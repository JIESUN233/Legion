call vcvars64.bat
cl /EHsc trie_st.cpp /O2
cl /EHsc trie_mt.cpp /O2
nvcc -I../include -arch=compute_70 -O2 trie.cu --expt-relaxed-constexpr --extended-lambda -Xcompiler /Zc:__cplusplus -o trie
