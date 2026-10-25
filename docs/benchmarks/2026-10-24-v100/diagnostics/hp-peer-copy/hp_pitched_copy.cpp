// Standalone transfer diagnostic. No kernels, model pointers, or device reset.
#include <cuda.h>
#include <dlfcn.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
constexpr size_t NG=8, VS=16160, V=129280, MAXT=64;
constexpr size_t SP=VS*4, DP=V*4;
const auto deadline=Clock::now()+std::chrono::seconds(55);
void budget(){if(Clock::now()>deadline)throw std::runtime_error("55s cooperative deadline");}
void ck(CUresult x,const char*what){budget();if(x!=CUDA_SUCCESS)throw std::runtime_error(std::string(what)+" CUDA code "+std::to_string(int(x)));}
struct Memory{unsigned long long total,free,used;};
struct Nvml {
 void*lib=nullptr; using Dev=void*;
 int(*init)()=nullptr;int(*shutdown)()=nullptr;int(*getdev)(unsigned,Dev*)=nullptr;int(*mem)(Dev,Memory*)=nullptr;
 std::array<Dev,NG> dev{};bool ready=false;
 template<class T> void sym(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));if(!f)throw std::runtime_error(std::string("missing NVML symbol ")+n);}
 void open(){lib=dlopen("libnvidia-ml.so.1",RTLD_NOW);if(!lib)throw std::runtime_error("NVML unavailable");sym(init,"nvmlInit_v2");sym(shutdown,"nvmlShutdown");sym(getdev,"nvmlDeviceGetHandleByIndex_v2");sym(mem,"nvmlDeviceGetMemoryInfo");if(init())throw std::runtime_error("NVML init failed");ready=true;for(unsigned i=0;i<NG;i++)if(getdev(i,&dev[i]))throw std::runtime_error("NVML get device failed");}
 void guard(unsigned long long minimum,const char*phase){budget();for(size_t i=0;i<NG;i++){Memory m{};if(mem(dev[i],&m))throw std::runtime_error("NVML memory query failed");std::printf("{\"kind\":\"memory\",\"phase\":\"%s\",\"gpu\":%zu,\"free_bytes\":%llu,\"minimum_bytes\":%llu}\n",phase,i,m.free,minimum);if(m.free<minimum)throw std::runtime_error("free memory admission failed GPU "+std::to_string(i));}}
 ~Nvml(){if(ready)shutdown();if(lib)dlclose(lib);}
};
struct Owned {
 std::array<CUcontext,NG> ctx{};std::array<CUdevice,NG> dev{};std::array<CUdeviceptr,NG> src{};
 CUdeviceptr dst=0;CUstream stream=nullptr;CUevent start=nullptr,stop=nullptr;
 ~Owned(){
  if(ctx[0]){cuCtxSetCurrent(ctx[0]);if(stream)cuStreamSynchronize(stream);if(start)cuEventDestroy(start);if(stop)cuEventDestroy(stop);if(stream)cuStreamDestroy(stream);if(dst)cuMemFree(dst);}
  for(size_t i=0;i<NG;i++)if(ctx[i]){cuCtxSetCurrent(ctx[i]);if(src[i])cuMemFree(src[i]);cuDevicePrimaryCtxRelease(dev[i]);}
 }
};
uint8_t pattern(size_t g,size_t row,size_t byte,unsigned rep){uint64_t x=byte+0x9e3779b97f4a7c15ULL*(row+1)+0xd1b54a32d192ed03ULL*(g+1)+0x94d049bb133111ebULL*(rep+1);x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;return uint8_t(x^(x>>32));}
int main(){
 std::setvbuf(stdout,nullptr,_IOLBF,0);
 try {
  Nvml nv;nv.open();nv.guard(1ULL<<30,"initial-before-contexts");
  ck(cuInit(0),"init");int count=0;ck(cuDeviceGetCount(&count),"device count");if(count!=8)throw std::runtime_error("requires exactly eight devices");
  Owned own;
  for(size_t i=0;i<NG;i++){ck(cuDeviceGet(&own.dev[i],int(i)),"device");int pitch=0;ck(cuDeviceGetAttribute(&pitch,CU_DEVICE_ATTRIBUTE_MAX_PITCH,own.dev[i]),"max pitch");if(pitch<0 || size_t(pitch)<DP)throw std::runtime_error("device max pitch too small");ck(cuDevicePrimaryCtxRetain(&own.ctx[i],own.dev[i]),"retain own process primary context");nv.guard(512ULL<<20,"after-context");}
  for(size_t i=0;i<NG;i++){ck(cuCtxSetCurrent(own.ctx[i]),"bind allocation");ck(cuMemAlloc(&own.src[i],MAXT*SP),"source allocate");nv.guard(512ULL<<20,"after-source-allocation");}
  ck(cuCtxSetCurrent(own.ctx[0]),"bind GPU0");ck(cuMemAlloc(&own.dst,MAXT*DP),"destination allocate");nv.guard(512ULL<<20,"after-destination-allocation");
  for(size_t i=1;i<NG;i++){auto e=cuCtxEnablePeerAccess(own.ctx[i],0);if(e!=CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)ck(e,"GPU0 peer access");nv.guard(512ULL<<20,"after-peer-enable");}
  ck(cuStreamCreate(&own.stream,CU_STREAM_NON_BLOCKING),"stream");nv.guard(512ULL<<20,"after-stream");
  ck(cuEventCreate(&own.start,0),"start event");nv.guard(512ULL<<20,"after-start-event");ck(cuEventCreate(&own.stop,0),"stop event");nv.guard(512ULL<<20,"after-events");
  std::array<std::vector<uint8_t>,NG> host;for(auto&x:host)x.resize(MAXT*SP);std::vector<uint8_t> out(MAXT*DP);
  unsigned long long verified=0;
  std::printf("{\"kind\":\"header\",\"source_payload_bytes\":%zu,\"destination_payload_bytes\":%zu,\"context_overhead_in_payload\":false,\"repeats\":20,\"cuda_kernel_launches\":0}\n",NG*MAXT*SP,MAXT*DP);
  for(size_t t: {1UL,3UL,12UL,64UL})for(unsigned rep=0;rep<20;rep++){
   budget();
   // Fresh deterministic bytes and synchronous HtoD are outside the event interval.
   for(size_t g=0;g<NG;g++){
    for(size_t row=0;row<t;row++)for(size_t b=0;b<SP;b++)host[g][row*SP+b]=pattern(g,row,b,rep);
    ck(cuCtxSetCurrent(own.ctx[g]),"bind upload");ck(cuMemcpyHtoD(own.src[g],host[g].data(),t*SP),"refresh source");ck(cuCtxSynchronize(),"source readiness");
   }
   ck(cuCtxSetCurrent(own.ctx[0]),"bind collect");
   // Poison destination outside timing so untouched regions cannot accidentally pass.
   ck(cuMemsetD8(own.dst,0xa5,t*DP),"destination poison");ck(cuCtxSynchronize(),"poison readiness");
   auto wall=Clock::now();ck(cuEventRecord(own.start,own.stream),"start record");
   for(size_t g=0;g<NG;g++){
    if(g==0){CUDA_MEMCPY3D c{};c.srcMemoryType=CU_MEMORYTYPE_DEVICE;c.srcDevice=own.src[0];c.srcPitch=SP;c.dstMemoryType=CU_MEMORYTYPE_DEVICE;c.dstDevice=own.dst;c.dstPitch=DP;c.WidthInBytes=SP;c.Height=t;c.Depth=1;ck(cuMemcpy3DAsync(&c,own.stream),"local pitched copy");}
    else{CUDA_MEMCPY3D_PEER c{};c.srcMemoryType=CU_MEMORYTYPE_DEVICE;c.srcDevice=own.src[g];c.srcContext=own.ctx[g];c.srcPitch=SP;c.dstMemoryType=CU_MEMORYTYPE_DEVICE;c.dstDevice=own.dst;c.dstContext=own.ctx[0];c.dstPitch=DP;c.dstXInBytes=g*SP;c.WidthInBytes=SP;c.Height=t;c.Depth=1;ck(cuMemcpy3DPeerAsync(&c,own.stream),"peer pitched copy");}
   }
   ck(cuEventRecord(own.stop,own.stream),"stop record");auto enqueued=Clock::now();ck(cuEventSynchronize(own.stop),"copy completion");auto complete=Clock::now();float ms=0;ck(cuEventElapsedTime(&ms,own.start,own.stop),"event elapsed");
   ck(cuMemcpyDtoH(out.data(),own.dst,t*DP),"verification download");
   for(size_t row=0;row<t;row++)for(size_t g=0;g<NG;g++)for(size_t b=0;b<SP;b++)if(out[row*DP+g*SP+b]!=host[g][row*SP+b])throw std::runtime_error("collected byte mismatch t="+std::to_string(t)+" repeat="+std::to_string(rep)+" row="+std::to_string(row)+" GPU="+std::to_string(g)+" byte="+std::to_string(b));
   verified+=t*DP;
   std::printf("{\"kind\":\"sample\",\"rows\":%zu,\"repeat\":%u,\"collected_bytes\":%zu,\"remote_bytes\":%zu,\"copy_event_ms\":%.6f,\"host_enqueue_ms\":%.6f,\"host_completion_ms\":%.6f,\"all_bytes_verified\":true}\n",t,rep,t*DP,7*t*SP,ms,std::chrono::duration<double,std::milli>(enqueued-wall).count(),std::chrono::duration<double,std::milli>(complete-wall).count());
  }
  nv.guard(512ULL<<20,"completed-before-release");
  std::printf("{\"kind\":\"success\",\"collections\":80,\"verified_bytes\":%llu,\"owned_resources_release_on_return\":true}\n",verified);
  return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"benchmark failed: %s\n",e.what());return 1;}
}
