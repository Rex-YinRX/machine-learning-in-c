if not exist "build-bat\bin" mkdir "build-bat\bin"
if not exist "build-bat\bin" exit /b 1

gcc -Iinclude -std=c11 -Wall -Wextra -Wpedantic -O2 -g programs/main.c src/matrix.c src/layer.c src/network.c -o "build-bat\bin\matrix_demo.exe"
gcc -Iinclude -std=c11 -Wall -Wextra -Wpedantic -O2 -g programs/load_model.c src/matrix.c src/layer.c src/network.c -o "build-bat\bin\load_model.exe"
gcc -Iinclude -std=c11 -Wall -Wextra -g tools/mnist.c src/matrix.c -o "build-bat\bin\mnist.exe"  
gcc -Iinclude -std=c11 -Wall -Wextra -O2 programs/digit_app.c programs/digit_preprocess.c src/matrix.c src/layer.c src/network.c -I.\raylib\include -L.\raylib\lib -lraylib -lopengl32 -lgdi32 -lwinmm -o "build-bat\bin\digit_app.exe"
