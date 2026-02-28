import pycuda.driver as drv
import run
from run import start_compact_sceme, three_greeds
from graphics import relative_errors_and_local_orders_graphic

if __name__ == '__main__':
    # start_compact_sceme()
    # three_greeds()
    relative_errors_and_local_orders_graphic()