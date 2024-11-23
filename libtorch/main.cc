#include <torch/script.h>

#include <iostream>
#include <memory>
using namespace std;

int main(int argc, const char* argv[]){                                     // --- (1)
    if (argc != 2) {
        cerr << "usage: example-app <path-to-exported-script-module>\n";    // --- (2)
    }

    torch::jit::script::Module module;                                      // --- (3)
    try {
        module = torch::jit::load(argv[1]);                                 // --- (4)
    } catch (const c10::Error& e){
        cerr<<"error loading the module \n";                                // --- (5)
        return -1;
    }

    cout << "ok \n";                                                        // --- (6)
}
