import pycuda.driver as drv


if __name__ == '__main__':
    pass
#     drv.init()
#
#     print('CUDA device query (PyCUDA version)')
#
#     print(f'Detected {drv.Device.count()} CUDA Capable device(s)')
#
#     for i in range(drv.Device.count()):
#         gpu_device = drv.Device(i)
#         print('Device {}: {}'.format(i, gpu_device.name()))
#
#         compute_capability = float('{}.{}'.format(*gpu_device.compute_capability()))
#         print('\t Compute Capability: {}'.format(compute_capability))
#         print('\t Total Memory: {} megabytes'.format(gpu_device.total_memory() // (1024 ** 2)))
#
#         # Get all device attributes
#         device_attributes_tuples = gpu_device.get_attributes().items()
#         device_attributes = {}
#
#         for k, v in device_attributes_tuples:
#             device_attributes[str(k)] = v
#
#         num_mp = device_attributes['MULTIPROCESSOR_COUNT']
#
#         # Cores per multiprocessor lookup table
#         cuda_cores_per_mp = {
#             5.0: 128, 5.1: 128, 5.2: 128,
#             6.0: 64, 6.1: 128, 6.2: 128,
#             7.0: 64, 7.2: 64, 7.5: 64,
#             8.0: 64, 8.6: 128, 8.9: 128,
#             9.0: 128
#         }.get(compute_capability, 0)  # Default to 0 if CC not found
#
#         print('\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(
#             num_mp, cuda_cores_per_mp, num_mp * cuda_cores_per_mp))
#
#         device_attributes.pop('MULTIPROCESSOR_COUNT')
#
#         for k in device_attributes.keys():
#             print('\t {}: {}'.format(k, device_attributes[k]))
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
