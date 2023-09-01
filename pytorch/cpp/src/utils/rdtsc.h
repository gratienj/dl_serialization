#ifndef TIME_STAMP_H
#define TIME_STAMP_H

static inline void rdtsc(volatile unsigned long long int *counter){
#ifndef WIN32
  asm volatile ("rdtsc \n\t"
      "movl %%eax,%0 \n\t"
      "movl %%edx,%1 \n\t"
      : "=m" (((unsigned *)counter)[0]), "=m" (((unsigned *)counter)[1])
      :
      : "eax" , "edx");
#endif  
}

#define RDTSC(X) asm volatile ("rdtsc \n\t"\
		       "movl %%eax,%0 \n\t"\
		       "movl %%edx,%1 \n\t"\
		       : "=m" (((unsigned *)(X))[0]), "=m" (((unsigned *)(X))[1])\
		       :\
		       : "eax" , "edx")
#endif
