"""Use loader order, not wgpu's unrelated adapter order, for TI/QD selection."""
from dataclasses import dataclass


@dataclass
class Device:
    index: int
    name: str
    fp64: bool
    memory: int

    def __str__(self):
        return self.name


def devices():
    import vulkan as vk
    app = vk.VkApplicationInfo(pApplicationName='gate0', apiVersion=vk.VK_API_VERSION_1_0)
    instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
    result = []
    try:
        for i, handle in enumerate(vk.vkEnumeratePhysicalDevices(instance)):
            props = vk.vkGetPhysicalDeviceProperties(handle)
            features = vk.vkGetPhysicalDeviceFeatures(handle)
            memory = vk.vkGetPhysicalDeviceMemoryProperties(handle)
            total = sum(memory.memoryHeaps[j].size for j in range(memory.memoryHeapCount)
                        if memory.memoryHeaps[j].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            result.append(Device(i, str(props.deviceName), bool(features.shaderFloat64), total))
    finally:
        vk.vkDestroyInstance(instance, None)
    return result
