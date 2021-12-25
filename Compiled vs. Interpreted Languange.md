# Compile and Debug
## Compile
- Compile is the process of turning code into machine instructions (or some kind of intermediate language, or bytecode, etc). A tool that does this is called a compiler.
- Source: https://www.computerhope.com/jargon/c/compile.htm
- Compile is the creation of an executable program from code written in a compiled programming language. Compiling allows the computer to run and understand the program without the need of the programming software used to create it. When a program is compiled it is often compiled for a specific platform (e.g., IBM platform) that works with IBM compatible computers, but not other platforms (e.g., Apple platform).
### Compiler
- Source: https://en.wikipedia.org/wiki/Compiler
- In computing, a compiler is a computer program that translates computer code written in one programming language (the source language) into another language (the target language). The name "compiler" is primarily used for programs that translate source code from a high-level programming language to a lower level language (e.g. assembly language, object code, or machine code) to create an executable program.
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