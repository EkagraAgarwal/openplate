"""Float32 Vulkan/WebGPU sweep using a WGSL compute shader."""
import numpy as np
from common import benchmark, parser, select, failure

SHADER = '''
struct Params { n:u32, color:u32, apply:u32, unused:u32 };
@group(0) @binding(0) var<storage,read_write> u:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read> k:array<f32>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let t=gid.x+gid.y*p.unused; let n=p.n; if(t>=n*n*n) {return;}
 let z=t%n; let j=(t/n)%n; let i=t/(n*n);
 if(p.apply==0u && (i+j+z)%2u!=p.color) {return;}
 var diagonal=0.0; var neighbors=0.0;
 for(var axis=0u;axis<3u;axis++) {
  let coord=select(select(z,j,axis==1u),i,axis==0u);
  let stride=select(select(1u,n,axis==1u),n*n,axis==0u);
  for(var d=-1;d<=1;d+=2) {
   if(i32(coord)+d<0 || i32(coord)+d>=i32(n)) {
    diagonal+=2.0*k[t]*f32(n*n);
   } else {
    let q=u32(i32(t)+d*i32(stride));
    let w=2.0*k[t]*k[q]/(k[t]+k[q])*f32(n*n);
    diagonal+=w; neighbors+=w*u[q];
   }
  }
 }
 if(p.apply!=0u) {out[t]=diagonal*u[t]-neighbors;}
 else {u[t]=(b[t]+neighbors)/diagonal;}
}
'''


def devices():
    import wgpu
    return [a for a in wgpu.gpu.enumerate_adapters_sync()
            if str(a.info['backend_type']).lower() == 'vulkan'
            and str(a.info['adapter_type']).lower() != 'cpu']


class WebGPU:
    def __init__(self, adapter):
        import wgpu
        self.wgpu = wgpu
        self.name = str(adapter.info['device'])
        limits = adapter.limits
        self.device = adapter.request_device_sync(required_limits={
            'max-storage-buffer-binding-size': limits['max-storage-buffer-binding-size'],
            'max-buffer-size': limits['max-buffer-size']})
        self.memory = 2**63
        self.module = self.device.create_shader_module(code=SHADER)
        self.pipeline = self.device.create_compute_pipeline(layout='auto', compute={
            'module': self.module, 'entry_point': 'main'})

    def execute(self, u, b, k, sweeps=0, reverse=False):
        wgpu, device = self.wgpu, self.device
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        buffers = [device.create_buffer_with_data(data=np.ascontiguousarray(a), usage=usage)
                   for a in (u, b, k)]
        buffers.append(device.create_buffer(size=u.nbytes, usage=usage))
        operations = [(c, 0) for _ in range(sweeps)
                      for c in ((1, 0) if reverse else (0, 1))] if sweeps else [(0, 1)]
        encoder = device.create_command_encoder()
        params = []
        groups = (u.size+127)//128
        columns = min(groups, 32768)
        for color, apply in operations:
            uniform = device.create_buffer_with_data(
                data=np.array([len(u), color, apply, columns*128], dtype=np.uint32),
                usage=wgpu.BufferUsage.UNIFORM)
            params.append(uniform)
            entries = [dict(binding=i, resource={'buffer': buf})
                       for i, buf in enumerate(buffers+[uniform])]
            group = device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0),
                                            entries=entries)
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self.pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(columns, (groups+columns-1)//columns)
            compute.end()
        device.queue.submit([encoder.finish()])
        data = device.queue.read_buffer(buffers[0 if sweeps else 3])
        result = np.frombuffer(data, dtype=np.float32).reshape(u.shape).copy()
        for buffer in buffers+params:
            buffer.destroy()
        return result

    def apply(self, u, k):
        return self.execute(u, u, k)

    def smooth(self, u, b, k, sweeps, reverse):
        return self.execute(u, b, k, sweeps, reverse)


if __name__ == '__main__':
    args = parser(False).parse_args()
    try:
        ds = devices()
        if args.list:
            for i, a in enumerate(ds):
                print(f'{i}: {dict(a.info)}')
        else:
            # Name selectors are resolved explicitly rather than adapter repr.
            a = ds[int(args.device)] if args.device.isdecimal() else select(
                [str(a.info['device']) for a in ds], args.device)
            if isinstance(a, str):
                a = next(d for d in ds if str(d.info['device']) == a)
            raise SystemExit(benchmark('wgpu', WebGPU(a), args, False, True))
    except Exception as exc:
        failure('wgpu', args.device, exc)
        raise SystemExit(1)
