# OS (Operating System)
- Examples: Windows, Linux, MacOS, Android, iOS
- Privileged Instruction: 시스템 요소들과 소통하는 명령.
- 하나의 하드웨어 시스템당 OS는 1개만 돌아갈 수 있음.
- 일반 프로그램들은 특권 명령이 필요 없어 여러 개를 동시에 수행 가능
## DOS (Disk Operating System)
## UNIX-based Operating Systems
### Linux
#### CentOS (Community ENTerprise Operating System)
- Source: https://en.wikipedia.org/wiki/CentOS
- CentOS (/ˈsɛntɒs/) is a Linux distribution that provides a free and open-source community-supported computing platform.
### MacOS
## Microsoft Windows
## Ubuntu

# SoC (System on Chip)
- Source: https://en.wikipedia.org/wiki/System_on_a_chip
- ***A system on a chip is an integrated circuit (also known as a "chip") that integrates all or most components of a computer or other electronic system.*** These components almost always include a central processing unit (CPU), memory, input/output ports and secondary storage, often alongside other components such as radio modems and a graphics processing unit (GPU) – all on a single substrate or microchip.

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

# Compile & Debug
## Compile
- Compile is the process of turning code into machine instructions (or some kind of intermediate language, or bytecode, etc). A tool that does this is called a compiler.
- Source: https://www.computerhope.com/jargon/c/compile.htm
- Compile is the creation of an executable program from code written in a compiled programming language. Compiling allows the computer to run and understand the program without the need of the programming software used to create it. When a program is compiled it is often compiled for a specific platform (e.g., IBM platform) that works with IBM compatible computers, but not other platforms (e.g., Apple platform).
### Compiler
- Source: https://en.wikipedia.org/wiki/Compiler
- In computing, a compiler is a computer program that translates computer code written in one programming language (the source language) into another language (the target language). The name "compiler" is primarily used for programs that translate source code from a high-level programming language to a lower level language (e.g. assembly language, object code, or machine code) to create an executable program.
### Assembler
### Compliled Language
- Source: https://www.geeksforgeeks.org/difference-between-compiled-and-interpreted-language/
- A compiled language is a programming language whose implementations are typically compilers and not interpreters.
- In this language, once the program is compiled it is expressed in the instructions of the target machine.
- There are at least two steps to get from source code to execution.
- Compiled programs run faster than interpreted programs and deliver better performance.
- Compilation errors prevent the code from compiling.
- The code of compiled language can be executed directly by the computer’s CPU
- Examples: C, C++, C#, CLEO, COBOL, etc.
### Interpreted Language
- Source: https://www.geeksforgeeks.org/difference-between-compiled-and-interpreted-language/
- An interpreted language is a programming language whose implementations execute instructions directly and freely, without previously compiling a program into machine-language instructions.
- The instructions are not directly executed by the target machine.
- There is only one step to get from source code to execution.
- In this language, interpreted programs can be modified while the program is running.
- All the debugging occurs at run-time.
- A program written in an interpreted language is not compiled, it is interpreted.
- This language delivers relatively slower performance.
- Examples: JavaScript, Perl, Python, BASIC, etc.
### Assembly Language (or Assembler Language)
- Source: https://en.wikipedia.org/wiki/Assembly_language
- In computer programming, assembly language (or assembler language), sometimes abbreviated asm, is any low-level programming language in which there is a very strong correspondence between the instructions in the language and the architecture's machine code instructions. *Assembly language usually has one statement per machine instruction (1:1), but constants, comments, assembler directives, symbolic labels of, e.g., memory locations, registers, and macros are generally also supported.*
- *Assembly code is converted into executable machine code by a utility program referred to as an assembler. The conversion process is referred to as assembly, as in assembling the source code. The computational step when an assembler is processing a program is called assembly time.*
### Machine Code
- *In computer programming, machine code is any low-level programming language, consisting of machine language instructions, which are used to control a computer's central processing unit (CPU). Each instruction causes the CPU to perform a very specific task, such as a load, a store, a jump, or an arithmetic logic unit (ALU) operation on one or more units of data in the CPU's registers or memory.*
- Machine code is a strictly numerical language which is designed to run as fast as possible, and may be considered as the lowest-level representation of a compiled or assembled computer program or as a primitive and hardware-dependent programming language. While it is possible to write programs directly in machine code, managing individual bits and calculating numerical addresses and constants manually is tedious and error-prone. For this reason, programs are very rarely written directly in machine code in modern contexts, but may be done for low level debugging, program patching (especially when assembler source is not available) and assembly language disassembly.
- *The majority of practical programs today are written in higher-level languages or assembly language. The source code is then translated to executable machine code by utilities such as compilers, assemblers, and linkers, with the important exception of interpreted programs,[nb 1] which are not translated into machine code. However, the interpreter itself, which may be seen as an executor or processor performing the instructions of the source code, typically consists of directly executable machine code (generated from assembly or high-level language source code).*
- Machine code is by definition the lowest level of programming detail visible to the programmer, but internally many processors use microcode or optimise and transform machine code instructions into sequences of micro-ops. This is not generally considered to be a machine code.
- ***The source code is often transformed by an assembler or compiler into binary machine code that can be executed by the computer.***
## Debug
- Debug is the act of finding out where in the code the application is going wrong. (= Get rid of bugs)

# PyPy
- Source: https://en.wikipedia.org/wiki/PyPy
- PyPy is an alternative implementation of the Python programming language to CPython (which is the standard implementation). PyPy often runs faster than CPython because PyPy uses a just-in-time compiler. Most Python code runs well on PyPy except for code that depends on CPython extensions, which either does not work or incurs some overhead when run in PyPy.

# Markup Language
- Source: https://techterms.com/definition/markup_language
- A markup language is a computer language that uses tags to define elements within a document. It is human-readable, meaning markup files contain standard words, rather than typical programming syntax. While several markup languages exist, the two most popular are HTML and XML.
- Since both HTML and XML files are saved in a plain text format, they can be viewed in a standard text editor.
## HTML (HyperText Markup Language)
- HTML is a markup language used for creating webpages. The contents of each webpage are defined by HTML tags. Most elements require a beginning and end tag, with the content placed between the tags.
## XML (eXtensible Markup Language)
- ***XML is used for storing structured data, rather than formatting information on a page. While HTML documents use predefined tags (like the examples above), XML files use custom tags to define elements.*** For example, an XML file that stores information about computer models may include the following section:
	```xml
	<computer>
	  <manufacturer>Dell</manufacturer>
	  <model>XPS 17</model>
	  <components>
		<processor>2.00 GHz Intel Core i7</processor>
		<ram>6GB</ram>
		<storage>1TB</storage>
	  </components>
	</computer>
	```
- XML is called the "Extensible Markup Language" since custom tags can be used to support a wide range of elements. Each XML file is saved in a standard text format, which makes it easy for software programs to parse or read the data. Therefore, XML is a common choice for exporting structured data and for sharing data between multiple programs.
## Markdown
- Source: https://en.wikipedia.org/wiki/Markdown
- ***Markdown is a lightweight markup language for creating formatted text using a plain-text editor.***
- ***In computing, formatted text, as opposed to plain text, is digital text which has styling information beyond the minimum of semantic elements: colours, styles (boldface, italic), sizes, and special features in HTML (such as hyperlinks).***

# Programming Languange
## Scripting Language
- Source: https://en.wikipedia.org/wiki/Scripting_language
- A scripting language or script language is a programming language for a runtime system that automates the execution of tasks that would otherwise be performed individually by a human operator. Scripting languages are usually interpreted at runtime rather than compiled.
### Python
### JavaScript
### Bash
### Kotlin
### PHP: Hypertext Preprocessor
- Source: https://en.wikipedia.org/wiki/Scripting_language
- ***PHP is a general-purpose scripting language geared towards web development.***
- Source: https://www.freecodecamp.org/news/what-is-php-the-php-programming-language-meaning-explained/
- ***Another beautiful thing about PHP is that you can embed it in HTML.***
### PowerShell
- A scripting language for use on Microsoft Windows operating systems.
### Visual Basic for Applications
- An extension language specifically for Microsoft Office applications.

# Notebook
- ***Notebooks are a form of interactive computing, in which users write and execute code, visualize the results, and share insights.*** Typically, data scientists use notebooks for experiments and exploration tasks.
## Jupyter Notebook (Formerly known as the IPython (Interactive Python) Notebook)
- Source: https://ipython.org/notebook.html
- The IPython Notebook is now known as the Jupyter Notebook. It is an interactive computational environment, in which you can combine code execution, rich text, mathematics, plots and rich media.
- Source: https://medium.com/memory-leak/data-science-notebooks-a-primer-4af256c8f5c6
- ***Notebooks are represented as JSON documents. In turn, notebooks can interweave code with natural language markup and HTML.***
- ***The browser passes the code to a back-end “kernel,” which runs the code and returns the results to the client. The kernel can run locally or in the cloud.***
## Apache Zeppelin
- Source: https://www.cloudera.com/products/open-source/apache-hadoop/apache-zeppelin.html
- ***A completely open web-based notebook that enables interactive data analytics.***
- Apache Zeppelin is a new and incubating multi-purposed web-based notebook which brings data ingestion, data exploration, visualization, sharing and collaboration features to ***Hadoop and Spark***.
- Apache Zeppelin is a new and upcoming web-based notebook which brings data exploration, visualization, sharing and collaboration features to Spark.   ***It support Python, but also a growing list of programming languages such as Scala, Hive, SparkSQL, shell and markdown.***
- Also when you are done with your notebook and found some insight you want to share, you can easily create a report out of it and either print it or send it out.
## Google Colab
- Source: https://medium.com/memory-leak/data-science-notebooks-a-primer-4af256c8f5c6
- Data scientists/ML engineers share notebooks today, but it isn’t easy to do with open source Jupyter. In contrast, ***Google Colab emphasizes sharing as part of its functionality.*** Individuals thought the opportunity to do “remote pair programming” in a notebook could be useful, especially for senior leaders trying to help junior individuals on the team.

# JSON (JavaScript Object Notation)
- JSON is simply a notation for encoding common computer data types in a readable form.
- Source: https://www.w3schools.com/whatis/whatis_json.asp
- ***JSON is a lightweight format for storing and transporting data***.
- ***JSON is often used when data is sent from a server to a web page. A common use of JSON is to read data from a web server, and display the data in a web page.***
```json
{
"employees":[
    {"firstName":"John", "lastName":"Doe"},
    {"firstName":"Anna", "lastName":"Smith"},
    {"firstName":"Peter", "lastName":"Jones"}
]
}
```
	- Data is in name/value pairs.
	- Data is separated by commas.
	- Curly braces hold objects.
	- Square brackets hold arrays.

# Binary Prefixes
- The binary prefixes include kibi, mebi, gibi, tebi, pebi, exbi, zebi and yobi.
- One gibibyte equals 2^30 or 1,073,741,824 bytes.
- The International Electrotechnical Commission (IEC) created these prefixes in 1998. Prior to that, the metric prefixes in the International System of Units (SI) were used across the board to refer to both the decimal system's power-of-10 multipliers and the binary system's power-of-two multipliers. The prefixes used in the SI system include kilo, mega, giga, tera, peta, exa, zetta and yotta.

# Hard-Coding and Soft-Coding
## Hard-Coding
- Source: https://www.quora.com/What-does-hard-coding-mean-in-programming
- ***Hard-coding is the practice of embedding data directly into the source code of a program. Like you fix the size of the array instead of dynamic memory allocation. You put in static elements(It’s like static initialization’s). Hard-coding passwords mean putting non-encrypted plain text passwords and other secret data into the source code. A programmer should not hard-code unless needed otherwise.***

# Input/Output (I/O)
- Source: https://en.wikipedia.org/wiki/Input/output
- In computing, input/output (I/O, or informally io or IO) is the communication between an information processing system, such as a computer, and the outside world, possibly a human or another information processing system. Inputs are the signals or data received by the system and outputs are the signals or data sent from it. The term can also be used as part of an action; to "perform I/O" is to perform an input or output operation.
## IOPS (Input/Output operations Per Second)
- Source: https://en.wikipedia.org/wiki/IOPS
- Input/output operations per second (IOPS, pronounced eye-ops) is an input/output performance measurement used to characterize computer storage devices like HDD and SDD.

# Interface
## UI (User Interface)
### GUI (Graphical User Interface)
### CLI (Command-Line Interface)
- Source: https://www.techtarget.com/searchwindowsserver/definition/command-line-interface-CLI
- ***A command-line interface (CLI) is a text-based user interface (UI) used to run programs, manage computer files and interact with the computer.*** Command-line interfaces are also called command-line user interfaces, console user interfaces and character user interfaces.
#### Shell
- In computing, *a shell program provides access to an operating system's components. The shell gives users (or other programs) a way to get "inside" the system to run programs or manage configurations.* The shell defines the boundary between inside and outside.
- *Two well-known CLI shells are PowerShell for Windows and Bash for Linux and macOS.*
- *Microsoft Windows includes the Command Prompt app as well as the PowerShell application, both of which can be used to interact directly with the computer.*
- *Linux and other Unix-based operating systems usually provide the Bourne-Again Shell (bash) as the default shell.* Other shells, including the C shell, Z shell and others, can be configured as the default system shell.
## API (Application Programming Interface)
- Source: https://rapidapi.com/blog/api-vs-gui/
- ***APIs allow applications to interact and communicate with an external server using some simple commands.*** Using APIs, developers can create streamlined processes that don’t keep re-inventing the wheel or building functionalities that already in existence.
- ***While an API permits the communication between two programs, GUI allows interaction between a human and a computer program.***
### REST API (REpresentational State Transfer API)

# Batch File
- Source: https://www.techtarget.com/searchwindowsserver/definition/batch-file
- ***A batch file is a text file that contains a sequence of commands for a computer operating system. It's called a batch file because it batches (bundles or packages) into a single file a set of commands that would otherwise have to be presented to the system interactively from a keyboard one at a time. A batch file is usually created for command sequences for which a user has a repeated need. Commonly needed batch files are often delivered as part of an operating system.*** You initiate the sequence of commands in the batch file by simply entering the name of the batch file on a command line.
- ***In UNIX-based operating systems, a batch file is called a shell script.***
- ***In the Disk Operating System (DOS), a batch file has the file name extension ".BAT".***

# OA (Office Automation)
- Office automation refers to the varied computer machinery and software used to digitally create, collect, store, manipulate, and relay office information needed for accomplishing basic tasks.

# Code Refactoring
- Source: https://en.wikipedia.org/wiki/Code_refactoring
- In computer programming and software design, ***code refactoring is the process of restructuring existing computer code—changing the factoring—without changing its external behavior. Refactoring is intended to improve the design, structure, and/or implementation of the software (its non-functional attributes), while preserving its functionality. Potential advantages of refactoring may include improved code readability and reduced complexity; these can improve the source code's maintainability and create a simpler, cleaner, or more expressive internal architecture or object model to improve extensibility. Another potential goal for refactoring is improved performance; software engineers face an ongoing challenge to write programs that perform faster or use less memory.***

# Pseudocode
- Source: https://economictimes.indiatimes.com/definition/pseudocode
- *Pseudocode is an informal way of programming description that does not require any strict programming language syntax or underlying technology considerations. It is used for creating an outline or a rough draft of a program. Pseudocode summarizes a program’s flow, but excludes underlying details.* System designers write pseudocode to ensure that programmers understand a software project's requirements and align code accordingly.
- *Pseudocode is not an actual programming language. So it cannot be compiled into an executable program. It uses short terms or simple English language syntaxes to write code for programs before it is actually converted into a specific programming language.* This is done to identify top level flow errors, and understand the programming data flows that the final program is going to use. This definitely helps save time during actual programming as conceptual errors have been already corrected. Firstly, program description and functionality is gathered and then pseudocode is used to create statements to achieve the required results for a program. Detailed pseudocode is inspected and verified by the designer’s team or programmers to match design specifications. *Catching errors or wrong program flow at the pseudocode stage is beneficial for development as it is less costly than catching them later.* Once the pseudocode is accepted by the team, it is rewritten using the vocabulary and syntax of a programming language. *The purpose of using pseudocode is an efficient key principle of an algorithm. It is used in planning an algorithm with sketching out the structure of the program before the actual coding takes place.*

# Encoding
## Base64
- Source: https://base64.guru/learn/what-is-base64
- *Base64 is a encoding algorithm that allows you to transform any characters into an alphabet which consists of Latin letters, digits, plus, and slash. Thanks to it, you can convert Chinese characters, emoji, and even images into a “readable” string, which can be saved or transferred anywhere.*
- To figuratively understand why Base64 was invented, imagine that during a phone call Alice wants to send an image to Bob. The first problem is that she cannot simply describe how the image looks, because Bob needs an exact copy. In this case, *Alice may convert the image into the binary system and dictate to Bob the binary digits (bits), after that he will be able to convert them back to the original image.* The second problem is that the tariffs for phone calls are too expensive and dictate each byte *as 8 binary digits will last too long. To reduce costs, Alice and Bob agree to use a more efficient data transfer method by using a special alphabet, which replaces every “six digits” with one “letter”.*
- To realize the difference, check out a 5x5 image converted to binary digits: 010001 110100 100101 000110 001110 000011 011101 100001 000000 010000 000000 000001 000000 001111 000000 000000 000000 001111 111100 000000 000000 000000 000000 000000 000000 000010 110000 000000 000000 000000 000000 000000 000000 010000 000000 000001 000000 000000 000000 000010 000000 100100 010000 000001 000000 000011 001011
- Although the same image converted to Base64 looks like this: R0lGODdhAQABAPAAAP8AAAAAACwAAAAAAQABAAACAkQBADs
- I think the difference is obvious. *Even if you remove spaces or padding zeros from binary digits, the Base64 string will still be shorter.* I grouped bits only to show that each group meets each character of the Base64 string.