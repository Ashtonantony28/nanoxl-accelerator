"""CSR map of ctrl_top (docs/control.md §5): offsets, bit fields, error codes."""

ID_VALUE = 0x4E584C43        # "NXLC"
VERSION_VALUE = 0x00010000

ID = 0x000
VERSION = 0x004
CTRL = 0x008                 # [0] step_mode, [1] soft_reset (write 1; reads 1 while in progress)
STATUS = 0x00C
ERR_CODE = 0x010
ERR_INDEX = 0x014
ERR_W0 = 0x018               # ERR_W0..3 at 0x018, 0x01C, 0x020, 0x024
RING_BASE = 0x030
RING_SIZE = 0x034            # in 16-byte words, 24 bits
RING_HEAD = 0x038
DOORBELL = 0x03C             # tail; a write rings the doorbell
IRQ_EN = 0x040
IRQ_STATUS = 0x044           # W1C
IRQ_CODE = 0x048
END_COUNT = 0x04C
DBG_STEP = 0x050
DBG_LAST = 0x054
SEM_SEL = 0x058
SEM_VAL = 0x05C
PERF_CTRL = 0x060            # [0] clear perf counters, [1] sat_clr
PERF_CYCLES = 0x064
PERF_BUSY_DMA = 0x068
PERF_BUSY_ARR = 0x06C
PERF_BUSY_VEC = 0x070
PERF_RETIRED = 0x074
PERF_BYTES_LO = 0x078
PERF_BYTES_HI = 0x07C
SAT_BASE = 0x080             # SAT_i at 0x080 + 4 i, i < NSAT
NSAT = 4
SAT_NAMES = ("requant", "alu", "ln_z", "ln_out")

CTRL_STEP = 1 << 0
CTRL_SOFT_RESET = 1 << 1

ST_HALTED = 1 << 0
ST_QUEUES_IDLE = 1 << 1
ST_RING_EMPTY = 1 << 2
ST_BUSY = 1 << 3
ST_QIDLE_SHIFT = 4           # [6:4] queue idle DMA, ARR, VEC
ST_EBUSY_SHIFT = 8           # [10:8] engine busy
ST_RING_CFG_OK = 1 << 13
ST_SRST_BUSY = 1 << 14
ST_Q_OVERFLOW = 1 << 15

IRQ_END = 1 << 0
IRQ_CMD = 1 << 1
IRQ_ERR = 1 << 2

ERR_NONE, ERR_OPCODE, ERR_RESV, ERR_FIELD, ERR_FETCH, ERR_DMA, ERR_RING, ERR_ENGINE = range(8)
ERR_NAMES = {0: "none", 1: "OPCODE", 2: "RESV", 3: "FIELD", 4: "FETCH", 5: "DMA", 6: "RING", 7: "ENGINE"}

SEM_COUNT = 32
SEM_MAX = 255
QDEPTH = 8
