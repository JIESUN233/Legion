#include "Server.h"
#include <stdlib.h>

int main(int argc, char** argv){
    Server* server = NewGPUServer();
    server->Initialize(atoi(argv[1]), atoi(argv[2]));//gpu number, default 1; in memory, default true
    server->PreSc(atoi(argv[3]));//cache aggregate mode, default 0
    server->Run();
    server->Finalize();
}