# SoC (System on Chip)
- Source: https://en.wikipedia.org/wiki/System_on_a_chip
- A system on a chip is an integrated circuit (also known as a "chip") that integrates all or most components of a computer or other electronic system. These components almost always include a central processing unit (CPU), memory, input/output ports and secondary storage, often alongside other components such as radio modems and a graphics processing unit (GPU) – all on a single substrate or microchip.

# Semiconductor
## Memory Chip
## Logic Chip
### CPU (Central Processing Unit)
- Source: https://www.weka.io/blog/cpu-vs-gpu/
- At the heart of any and every computer in existence is a central processing unit or CPU. The CPU handles the core processing tasks in a computer—the literal computation that drives every single action in a computer system.
- Computers work through the processing of binary data, or ones and zeroes. To translate that information into the software, graphics, animations, and every other process executed on a computer, those ones and zeroes must work through the logical structure of the CPU. That includes the basic arithmetic, logical functions (AND, OR, NOT) and input and output operations. The CPU is the brain, taking information, calculating it, and moving it where it needs to go.
- CPUs cannot handle parallel processing like a GPU, so large tasks that require thousands or millions of identical operations will choke a CPU’s capacity to process data.
#### Core(s)
- The central architecture of the CPU is the “core,” where all computation and logic happens. A core typically functions through what is called the “instruction cycle,” where instructions are pulled from memory (fetch), decoded into processing language (decode), and executed through the logical gates of the core (execute). Initially, all CPUs were single-core, but with the proliferation of multi-core CPUs, we’ve seen an increase in processing power.
#### Cache
- Cache is super-fast memory built either within the CPU or in CPU-specific motherboards to facilitate quick access to data the CPU is currently using. Since CPUs work so fast to complete millions of calculations per second, they require ultra-fast (and expensive) memory to do it—memory that is much faster than hard drive storage or even the fastest RAM.
- In any CPU configuration, you will see some L1, L2, and/or L3 cache arrangement, with L1 being the fastest and L3 the slowest. The CPU will store the most immediately needed information in L1, and as the data loses priority, it will move out into L2, then L3, and then out to RAM or the hard disk.
#### Memory Management Unit (MMU)
- The MMU controls data movement between the CPU and RAM during the instruction cycle.
#### CPU Clock and Control Unit
- Every CPU works on synchronizing processing tasks through a clock. The CPU clock determines the frequency at which the CPU can generate electrical pulses, its primary way of processing and transmitting data, and how rapidly the CPU can work. So, the higher the CPU clock rate, the faster it will run and quicker processor-intensive tasks can be completed.
### GPU (Graphics Processing Unit)
- Source: https://www.weka.io/blog/cpu-vs-gpu/
- The challenge in processing graphics is that graphics call on complex mathematics to render, and those complex mathematics must compute in parallel to work correctly. For example, a graphically intense video game might contain hundreds or thousands of polygons on the screen at any given time, each with its individual movement, color, lighting, and so on. CPUs aren’t made to handle that kind of workload. That’s where graphical processing units (GPUs) come into play.
- GPUs are similar in function to CPU: they contain cores, memory, and other components. Instead of emphasizing context switching to manage multiple tasks, GPU acceleration emphasizes parallel data processing through a large number of cores.
- These cores are usually less powerful individually than the core of a CPU. GPUs also typically have less interoperability with different hardware APIs and houseless memory. Where they shine is pushing large amounts of processed data in parallel. Instead of switching through multiple tasks to process graphics, a GPU simply takes batch instructions and pushes them out at high volume to speed processing and display.
- A GPU consist of hundreds of cores performing the same operation on multiple data items in parallel. Because of that, a GPU can push vast volumes of processed data through a workload, speeding up specific tasks beyond what a CPU can handle.
- Whereas CPUs excel in more complex computations, GPUs excel in extensive calculations with numerous similar operations, such as computing matrices or modeling complex systems.
- Tasks
	- Multitasking: GPUs aren’t built for multitasking, so they don’t have much impact in areas like general-purpose computing.
	- Cost: While the price of GPUs has fallen somewhat over the years, they are still significantly more expensive than CPUs. This cost rises more when talking about a GPU built for specific tasks like mining or analytics.
	- Power and Complexity: While a GPU can handle large amounts of parallel computing and data throughput, they struggle when the processing requirements become more chaotic. Branching logic paths, sequential operations, and other approaches to computing impede the effectiveness of a GPU.
- Applications
	- Bitcoin Mining: The process of mining bitcoins involves using computational power to solve complex cryptographic hashes. The increasing expansion of Bitcoin and the difficulty of mining bitcoins has led bitcoin mines to implement a GPU to handle immense volumes of cryptographic data in the hopes of earning bitcoins.
	- Machine Learning: Neural networks, particularly those used for deep-learning algorithms, function through the ability to process large amounts of training data through smaller nodes of operations. GPUs for machine learning have emerged to help process the enormous data sets used to train machine-learning algorithms and AI.
	- Analytics and Data Science: GPUs are uniquely suited to help analytics programs process large amounts of base data from different sources. Furthermore, these same GPUs can power the computation necessary for deep data sets associated with research areas like life sciences (genomic sequencing).
	
# Filename Extension
- Source: https://en.wikipedia.org/wiki/Filename_extension
- A filename extension, file extension or file type is an identifier specified as a suffix to the name of a computer file (.txt, .docx, .ppt, etc.). The extension indicates a characteristic of the file contents or its intended use. A filename extension is typically delimited from the filename with a full stop (period).