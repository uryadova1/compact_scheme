import pycuda.driver as drv
from run import start_compact_sceme, three_greeds
from graphics import relative_errors_and_local_orders_graphic
from run import check

if __name__ == '__main__':
    check()
    # start_compact_sceme()
    # three_greeds()
    # relative_errors_and_local_orders_graphic()
