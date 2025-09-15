# To use this file, type "source path/to/compilation/linux.sh"

echo "Run \"build\" to compile maxpro.cpp using g++"
echo "(compiles into maxpro.so, so it can be used in the Jupyter notebook)"

build()
{
	g++ -shared -fPIC -o maxpro.so maxpro.cpp
}

bdebug()
{
	g++ -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -shared -fPIC -o maxpro.so maxpro.cpp
}
