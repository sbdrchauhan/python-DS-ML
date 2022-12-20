# Linux:

![linux](./images/linux.png)

Linux is widely popular and is available in many applications. One also needs to know how to navigate through these systems either using command lines or the GUI interface. But, command line is the most powerful tool. Here I just want to make notes on the commands or the tasks that might be of importance for one to know.

```bash
## System Basics:
#---------------------------------------------------------------------------
$ echo $PATH          # displays what PATH environment variable is pointing to
$ which ls            # shows the path of the file where this command lives
$ sudo locate *.h     # locates the file path for all .h files
$ find *.sh           # finds the file
$ find / -name *.sh   # finds inside / root folder
$ ls /path/to/list    # lists the file and folders
$ ls -ltr <dir>       # more information while listings
$ ls \                # waits for command next line, backslash is used for command continuation
> /                   # like doing: $ ls /
$ apropos copy        # to see if any commands that does similar thing is available
                      # apropos helps to find commands we might need
$ man cp              # after you found the command that might be useful
                      # learn more about it using man command to see it's manual page
$ cat /etc/*release   # shows about the system installed
$ hostnamectl         # more system info.
$ uname -a            # similar as above info.
$ uptime              # shows how long this system has been booted
$ df -h               # shows how much disk available in human-readable format
$ free                # shows the memory usage
$ top                 # to see which app is using cpus

$ echo h{a,e,i,o,u}llo  # prints all possibility
$ echo h{a..z}llo       # range from a-z
$ echo h{z..a}llo       # reverse range from z to a
$ echo h{0..10}llo      # range of numbers as well
$ echo {0..100..2}      # 0 through 100 interval 2

$ echo "It is " $(date) "today."  # $(date) gives todays datetime
$ history                         # to show bash command history
```