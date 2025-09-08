# To use this file, type "source path/to/compilation/linux.sh"

echo "Run \"build\" to compile maxpro.cpp using g++"
echo "(compiles into maxpro.so, so it can be used in the Jupyter notebook)"

build()
{
	g++ -shared -fPIC -o maxpro.so maxpro.cpp
}
