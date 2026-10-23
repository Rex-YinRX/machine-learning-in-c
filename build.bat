gcc -std=c11 -Wall -Wextra -Wpedantic -O2 -g main.c matrix.c layer.c network.c -o matrix_demo.exe
gcc -std=c11 -Wall -Wextra -Wpedantic -O2 -g load_model.c matrix.c layer.c network.c -o load_model.exe
gcc -std=c11 -Wall -Wextra -g .\data\mnist.c matrix.c -o .\data\mnist.exe  
gcc -std=c11 -Wall -Wextra -O2 .\digit_app.c .\digit_preprocess.c .\matrix.c .\layer.c .\network.c -I.\raylib\include -L.\raylib\lib -lraylib -lopengl32 -lgdi32 -lwinmm -o .\digit_app.exe
