# `apt` (Advanced Package Tool) (Linux)
- Source: https://www.geeksforgeeks.org/apt-get-command-in-linux-with-examples/
- `apt-get` is a command-line tool which helps in handling packages in Linux. Its main task is to retrieve the information and packages from the authenticated sources for installation, upgrade and removal of packages along with their dependencies.
## `apt-get`
### `apt-get update`
- ***This command is used to synchronize the package index files from their sources again. You need to perform an update before you upgrade or dist-upgrade.***
### `apt-get upgrade`
- ***This command is used to install the latest versions of the packages currently installed on the user’s system from the sources enumerated in `/etc/apt/sources.list`. You need to perform an update before the upgrade, so that apt-get knows that new versions of packages are available.***
### `apt-get install`
- This command is used to install or upgrade packages. It is followed by one or more package names the user wishes to install. All the dependencies of the desired packages will also be retrieved and installed. ***The user can also select the desired version by following the package name with an ‘equals’ and the desired version number. Both of these version selection methods have the potential to downgrade the packages, so must be used with care.***

## `pwd`: Print working directory
## `cd`: Change directory
## `mkdir`: Create a directory
## `rmdir`: Remove an empty directory
### `rm -R`: Remove nested directories
## `ls`
- Source: https://linuxize.com/post/how-to-list-files-in-linux-using-the-ls-command/
- When used with no options and arguments, ls displays a list of the names of all files in the current working directory.
```
ls [OPTIONS] [FILES]
```
- `[OPTIONS]`
	- `-l`: The default output of the `ls` command shows only the names of the files and directories, which is not very informative. ***The `-l` option tells ls to print files in a long listing format. When the long listing format is used, you can see the following file information: The file type, The file permissions, Number of hard links to the file, File owner, File group, File size, Date and Time, File name.***
	- `-a`: ***To display all files including the hidden files use the `-a` option.***
	- `-R`: ***The `-R` option tells the `ls` command to display the contents of the subdirectories recursively.***
## `open`: Open files
## `cp`:  Copy a file to another directory
## `mv`: Move a file
## `sudo`: Execute commands with superuser privileges
## `exit`
- This command will close out the current session in the Terminal.
