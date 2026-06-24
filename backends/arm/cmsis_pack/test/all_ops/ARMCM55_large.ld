/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Derived from the ARM::Cortex_DFP ARMCM55 GCC linker script
 * (Device/ARMCM55/Config/ARMCM55_gcc.ld), retargeted to the Corstone-300
 * (SSE-300) FVP's DDR region.
 *
 * The all-ops coverage build links EVERY ExecuTorch operator into one image
 * (~5-6 MB code + ~28 MB static pools), which does not fit the small on-chip
 * regions (ITCM 512 KB @0x10000000, DTCM 512 KB, SRAM 2 MB). The FVP backs a
 * 1.5 GB DDR region at 0x70000000 (rwx), so everything is placed there. This
 * image exists only to exercise the full compile+link+execute surface; no real
 * device runs all operators at once.
 *
 * Boot: the vector table is the first thing in FLASH (0x70000000), so run the
 * FVP with  -C cpu0.INITSVTOR=0x70000000  so reset fetches SP/PC from DDR.
 *
 *-------- <<< Use Configuration Wizard in Context Menu >>> -------------------
 */

/*---------------------- Flash Configuration (DDR window) ---------------------*/
__ROM_BASE = 0x70000000;   /* Corstone-300 FVP DDR, executable (rwx) */
__ROM_SIZE = 0x20000000;   /* 512 MB */

/*--------------------- Embedded RAM Configuration (DDR window) ---------------*/
/* RAM must stay below 0xA0000000. The Armv8-M default memory map marks
   0xA0000000-0xDFFFFFFF as Device memory, where EVERY unaligned access faults
   regardless of CCR.UNALIGN_TRP. The stack/heap sit at the top of RAM, and
   CLANG emits unaligned halfword/word stores (e.g. into the 1-byte-aligned
   kernel-key char buffer in make_kernel_key_string), which then trap. Keeping
   RAM in 0x90000000-0xA0000000 -- DDR-backed AND Normal memory -- avoids it. */
__RAM_BASE = 0x90000000;   /* DDR, Normal memory (0x80000000-0x9FFFFFFF) */
__RAM_SIZE = 0x10000000;   /* 256 MB; ends at 0xA0000000 (Device boundary) */

/*--------------------- Stack / Heap Configuration ----------------------------*/
__STACK_SIZE = 0x00010000; /* 64 KB */
__HEAP_SIZE  = 0x00100000; /* 1 MB  */

__STACKSEAL_SIZE = 0;


MEMORY
{
  FLASH (rx)  : ORIGIN = __ROM_BASE, LENGTH = __ROM_SIZE
  RAM   (rwx) : ORIGIN = __RAM_BASE, LENGTH = __RAM_SIZE
}

ENTRY(Reset_Handler)

SECTIONS
{
  .text :
  {
    KEEP(*(.vectors))
    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)

    KEEP(*(.eh_frame*))
  } > FLASH

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > FLASH

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > FLASH
  __exidx_end = .;

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;

    LONG (LOADADDR(.data))
    LONG (ADDR(.data))
    LONG (SIZEOF(.data) / 4)

    __copy_table_end__ = .;
  } > FLASH

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    __zero_table_end__ = .;
  } > FLASH

  /*
   * This __etext variable is kept for backward compatibility with older,
   * ASM based startup files.
   */
  PROVIDE(__etext = LOADADDR(.data));

  .data : ALIGN(4)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    /* preinit data.  picolibc's __libc_init_array walks a single combined
       array bounded by __bothinit_array_start / __bothinit_array_end (rather
       than GNU's separate preinit and init arrays), so mark that span here --
       otherwise those bounds are weak-undefined (0/0) and NO static ctors run
       under CLANG/picolibc. Harmless for GCC/newlib, which uses the init array. */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    PROVIDE_HIDDEN (__bothinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    /* init data -- kept contiguous with preinit for __bothinit_array */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
    PROVIDE_HIDDEN (__bothinit_array_end = .);

    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(4);
    /* All data end */
    __data_end__ = .;

  } > RAM AT > FLASH

  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > RAM AT > RAM

  /* TLS template: picolibc places errno (and other __thread vars) here. */
  .tdata : ALIGN(8)
  {
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  } > RAM AT > FLASH

  .tbss (NOLOAD) : ALIGN(8)
  {
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.* .tcommon)
    __tbss_end = .;
  } > RAM

  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > RAM

  .stack (ORIGIN(RAM) + LENGTH(RAM) - __STACK_SIZE - __STACKSEAL_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > RAM
  PROVIDE(__stack = __StackTop);

  /* picolibc crt0 (Arm Toolchain for Embedded) expects these symbols. They
     are unused by GCC/newlib and AC6, so PROVIDE keeps this script portable:
     the aliases only materialise when picolibc's startup references them. */
  PROVIDE(__data_start  = __data_start__);
  PROVIDE(__data_end    = __data_end__);
  PROVIDE(__data_source = LOADADDR(.data));
  PROVIDE(__data_size   = __data_end__ - __data_start__);
  PROVIDE(__bss_start   = __bss_start__);
  PROVIDE(__bss_end     = __bss_end__);
  PROVIDE(__bss_size    = __bss_end__ - __bss_start__);
  PROVIDE(__tls_base     = __tdata_start);
  PROVIDE(__tls_end      = __tbss_end);
  PROVIDE(__tdata_source = LOADADDR(.tdata));
  PROVIDE(__tdata_size   = __tdata_end - __tdata_start);
  PROVIDE(__tbss_size    = __tbss_end - __tbss_start);
  PROVIDE(__tbss_offset  = __tbss_start - __tdata_start);
  PROVIDE(__tls_size     = __tbss_end - __tdata_start);
  PROVIDE(__tls_align    = 8);
  PROVIDE(__arm32_tls_tcb_offset = 8);
  PROVIDE(__stack_limit = __StackLimit);
  PROVIDE(__heap_start  = __end__);
  PROVIDE(__heap_end    = __HeapLimit);

  /* Check if data + heap + stack exceeds RAM limit */
  ASSERT(__StackLimit >= __HeapLimit, "region RAM overflowed with stack")
}
