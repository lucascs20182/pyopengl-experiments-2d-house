import sys
import glm # Versão em Python da OpenGL Mathematics (GLM)
import OpenGL.GL as gl # Funcoes da API OpenGL
import OpenGL.GLUT as glut # Criacao de janelas acesso ao teclado
import numpy as np
import ctypes


# Variáveis globais
shaderProgram = None
VBO = [None] * 100  # inicializando Vertex Buffer Object
VAO = [None] * 100  # inicializando  Array Object

vertex_shader_codigo= """
#version 330 core
in vec3 vPos;
in vec3 vCor;

out vec3 color_vert;

uniform mat4 transformacao;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 proj_matrix;

void main()
{
    gl_Position = proj_matrix * view_matrix * model_matrix * transformacao * vec4(vPos.x, vPos.y, vPos.z, 1.0f);
    color_vert = vCor;
}
"""

fragment_shader_codigo= """
#version 330 core
in vec3 color_vert;
out vec4 FragColor;
void main()
{
    FragColor = vec4(color_vert, 1.0f);
}
"""


# criação dos vértices
# especificação da forma

def create_data_triangulo_1():
    
    v1 = glm.vec3(-0.9, -0.3, 0.0) 
    v2 = glm.vec3(0.1, -0.3, 0.0) 
    v3 = glm.vec3(-0.35, 0.3, 0.0) 
    tri = glm.mat3(v1, v2, v3) 
    return tri


def create_data_triangulo_2():
    
    v1 = glm.vec3(-0.9, -0.3, 0.0) 
    v2 = glm.vec3(0.1, -0.3, 0.0) 
    v3 = glm.vec3(-0.35, 0.3, 0.0) 
    tri = glm.mat3(v1, v2, v3) 
    return tri

def create_quad():

    lista_vertices = [
        -0.5, -0.5, -0.5, 
        0.5, -0.5, -0.5, 
        0.5,  0.5, -0.5,
        0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5, 
        -0.5, -0.5, -0.5 
        ]

    vertices = np.array(lista_vertices, dtype=np.float32)
    
    return vertices


def create_quad2():

    lista_vertices = [
        -0.5, -0.5, -0.5, 
        0.5, -0.5, -0.5, 
        0.5,  0.5, -0.5, 
        0.5,  0.5, -0.5, 
        -0.5,  0.5, -0.5,
        -0.5, -0.5, -0.5 
        ]

    vertices = np.array(lista_vertices, dtype=np.float32)
    
    return vertices


def create_telhado():

    lista_vertices = [
        -0.5, -0.5, -0.5, 
        0.5, -0.5, -0.5, 
        0.5,  0.5, -0.5, 
        0.5,  0.5, -0.5, 
        -0.5,  0.5, -0.5, 
        -0.5, -0.5, -0.5 
        ]

    vertices = np.array(lista_vertices, dtype=np.float32)
    
    return vertices

def create_janela():

    lista_vertices = [
        -0.5, -0.5, -0.5, 
        0.5, -0.5, -0.5, 
        0.5,  0.5, -0.5, 
        0.5,  0.5, -0.5, 
        -0.5,  0.5, -0.5, 
        -0.5, -0.5, -0.5 
        ]

    vertices = np.array(lista_vertices, dtype=np.float32)
    
    return vertices

def create_porta():

    lista_vertices = [
        -0.5, -0.5, -0.5, 
        0.5, -0.5, -0.5, 
        0.5,  0.5, -0.5, 
        0.5,  0.5, -0.5, 
        -0.5,  0.5, -0.5,
        -0.5, -0.5, -0.5 
        ]

    vertices = np.array(lista_vertices, dtype=np.float32)
    
    return vertices

# transforma a config do VAO e VBO em uma função
# para reutilizar código
# função usada para configurações do triangulo
def configura_VAO_e_VBO_triangulo(color_VBO, lista_cores, indiceVAO, data):
    gl.glBindVertexArray(VAO[indiceVAO])
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO[indiceVAO]) # Efetua o bind do VBO
    gl.glBufferData(target=gl.GL_ARRAY_BUFFER, size= glm.sizeof(data), data=glm.value_ptr(data), usage=gl.GL_STATIC_DRAW)
    local = gl.glGetAttribLocation(shaderProgram, 'vPos')
    vertexDim = 3 # quantidade de elementos do vetor declarado no shader
    stride = 0 # Espaço em bytes até o próximo valor. E.g. próximo x, quando for posição (X | Y | Z | X | Y | ...)
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local, vertexDim, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    colors_to_buffer = np.array(lista_cores, dtype=np.float32)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_VBO[indiceVAO]) # Efetua o bind do VBO
    gl.glBufferData(gl.GL_ARRAY_BUFFER,(ctypes.c_float*len(colors_to_buffer))(*colors_to_buffer),gl.GL_DYNAMIC_DRAW) # usando ctypes

    local_vCor = gl.glGetAttribLocation(shaderProgram, 'vCor')
    tam_cores = 3 # Quantidade valores que definem a cor (tipo vec3)
    stride = 0 # Espaço entre os dados de cor
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local_vCor, tam_cores, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local_vCor) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    # Desvincula o VAO, VBO
    gl.glBindVertexArray(0) # Importante: Unbind do VAO primeiro
    gl.glDisableVertexAttribArray(local)
    gl.glDisableVertexAttribArray(local_vCor)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


# transforma a config do VAO e VBO em uma função
# para reutilizar código
# função usada para configurações do quadrado
def configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, indiceVAO, data):
    gl.glBindVertexArray(VAO[indiceVAO])
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO[indiceVAO]) # Efetua o bind do VBO
    gl.glBufferData(gl.GL_ARRAY_BUFFER, (ctypes.c_float*len(data))(*data), gl.GL_STATIC_DRAW)
    local = gl.glGetAttribLocation(shaderProgram, 'vPos')
    vertexDim = 3 # quantidade de elementos do vetor declarado no shader
    stride = 0 # Espaço em bytes até o próximo valor. E.g. próximo x, quando for posição (X | Y | Z | X | Y | ...)
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local, vertexDim, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    colors_to_buffer = np.array(lista_cores, dtype=np.float32)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_VBO[indiceVAO]) # Efetua o bind do VBO
    gl.glBufferData(gl.GL_ARRAY_BUFFER,(ctypes.c_float*len(colors_to_buffer))(*colors_to_buffer),gl.GL_DYNAMIC_DRAW) # usando ctypes

    local_vCor = gl.glGetAttribLocation(shaderProgram, 'vCor')
    tam_cores = 3 # Quantidade valores que definem a cor (tipo vec3)
    stride = 0 # Espaço entre os dados de cor
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local_vCor, tam_cores, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local_vCor) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    # Desvincula o VAO, VBO
    gl.glBindVertexArray(0) # Importante: Unbind do VAO primeiro
    gl.glDisableVertexAttribArray(local)
    gl.glDisableVertexAttribArray(local_vCor)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

# cuida da posição do desenho, escala etc.
def display_triangulo(indiceVAO, translatarX, translatarY, translatarZ, escalarX, escalarY, escalarZ):
    # Triangulo 1
    gl.glBindVertexArray(VAO[indiceVAO])

    # Cria matrizes de transformação
    matriz_transformacao = glm.identity(glm.mat4) # Cria matriz identidade
    # Quanto eu quero transladar em cada eixo
    vetor_translacao = glm.vec3(translatarX, translatarY, translatarZ)
    matriz_transformacao = glm.translate(matriz_transformacao, vetor_translacao)

    # Aumentar o tamanho do quadrado (escala)
    #escalar = 2.5
    # Aplica um mesmo valor de escala em todos os eixos
    vetor_escala = glm.vec3(escalarX, escalarY, escalarZ)
    identidade = glm.mat4(1.0)
    matriz_escala = glm.scale(identidade, vetor_escala)

    matriz_transformacao = matriz_transformacao * matriz_escala

    # Localizacao da variavel Uniform matriz transformacao no vertex shader
    transformacao_loc = gl.glGetUniformLocation(shaderProgram, "transformacao")
    # Copia os dados da matriz de transformacao para matriz Uniform dentro do vertex shader
    gl.glUniformMatrix4fv(transformacao_loc, 1, gl.GL_FALSE, glm.value_ptr(matriz_transformacao))

    quant_vertices = 3 
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, quant_vertices)
    gl.glBindVertexArray(0) # Desvincula o VAO



# cuida da posição do quadrado, escala do quadrado etc.
def display_quadrado(indiceVAO, translatarX, translatarY, translatarZ, escalarX, escalarY, escalarZ):
    gl.glBindVertexArray(VAO[indiceVAO])

    # Cria matrizes de transformação
    matriz_transformacao = glm.identity(glm.mat4) # Cria matriz identidade
    # Quanto eu quero transladar em cada eixo
    vetor_translacao = glm.vec3(translatarX, translatarY, translatarZ)
    matriz_transformacao = glm.translate(matriz_transformacao, vetor_translacao)

    # Aumentar o tamanho do quadrado (escala)
    #escalar = 3.0
    # Aplica um mesmo valor de escala em todos os eixos
    vetor_escala = glm.vec3(escalarX, escalarY, escalarZ)
    identidade = glm.mat4(1.0)
    matriz_escala = glm.scale(identidade, vetor_escala)

    matriz_transformacao = matriz_transformacao * matriz_escala

    # Localizacao da variavel Uniform matriz transformacao no vertex shader
    transformacao_loc = gl.glGetUniformLocation(shaderProgram, "transformacao")
    # Copia os dados da matriz de transformacao para matriz Uniform dentro do vertex shader
    gl.glUniformMatrix4fv(transformacao_loc, 1, gl.GL_FALSE, glm.value_ptr(matriz_transformacao))

    quant_vertices = 6
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, quant_vertices)
    gl.glBindVertexArray(0) # Desvincula o VAO



def create_buffers(data_t1, data_t2, data_q1, data_q2, data_q3, data_q4, 
    data_q5, data_q6, data_q7, data_q8, data_q9):
    global VAO
    global VBO
    color_VBO = [None] * 100

    count = 0
    while(count <= 20):
        VAO[count] = gl.glGenVertexArrays(1)
        count += 1
    

    count = 0
    while(count <= 20):
        VBO[count] = gl.glGenBuffers(1)
        count += 1


    count = 0
    while(count <= 20):
        color_VBO[count] = gl.glGenBuffers(1)
        count += 1
    

    # Triangulo 1
    lista_cores = [
    1.0, 0.3, 0.7, 
    1.0, 0.3, 0.7, 
    1.0, 0.3, 0.7 
    ] 

    configura_VAO_e_VBO_triangulo(color_VBO, lista_cores, 0, data_t1)



    # parte triangular do telhado
    lista_cores = [
    1, 0.3, 0.5, 
    1, 0.3, 0.5, 
    1, 0.3, 0.5 
    ] 
    
    configura_VAO_e_VBO_triangulo(color_VBO, lista_cores, 1, data_t2)




    # parte retangular do telhado
    lista_cores = [
    1, 0.3, 0.5, 
    1, 0.3, 0.5, 
    1, 0.3, 0.5,
    1, 0.3, 0.5, 
    1, 0.3, 0.5, 
    1, 0.3, 0.5 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 4, data_q3)



    # Quadrado 1
    lista_cores = [
    0.9, 0.7, 0.6, 
    0.9, 0.7, 0.6, 
    0.9, 0.7, 0.6,
    0.9, 0.7, 0.6, 
    0.9, 0.7, 0.6, 
    0.9, 0.7, 0.6 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 2, data_q1)



    # Janela 1: tá tendo uma rave, não tira ponto.
    lista_cores = [
        0.3, 0, 0, 1.0, 
        0.3, 0, 0, 1.0, 
        0.3, 0, 0, 1.0,
        0.3, 0, 0, 1.0, 
        0.3, 0, 0, 1.0, 
        0.3, 0, 0, 1.0 
    ]

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 5, data_q4)
    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 5, data_q4) # Janela 2

    

    # cor do sol
    lista_cores = [
    1.0, 1.0, 0.0, 
    1.0, 1.0, 0.0, 
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0, 
    1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 7, data_q6)



    # porta 1
    lista_cores = [
    0, 0.9, 0.6, 
    0, 0.9, 0.6, 
    0, 0.9, 0.6,
    0, 0.9, 0.6, 
    0, 0.9, 0.6, 
    0, 0.9, 0.6 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 6, data_q5)


    # Quadrado 2
    lista_cores = [
    0.9, 0.7, 0.7, 
    0.9, 0.7, 0.7, 
    0.9, 0.7, 0.7,
    0.9, 0.7, 0.7, 
    0.9, 0.7, 0.7, 
    0.9, 0.7, 0.7 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 3, data_q2)
    


    # gramado
    lista_cores = [
    0.0, 0.5, 0.0 , 
    0.0, 0.5, 0.0 , 
    0.0, 0.5, 0.0 ,
    0.0, 0.5, 0.0 , 
    0.0, 0.5, 0.0 , 
    0.0, 0.5, 0.0  
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 8, data_q3)


    # nuvem 1
    lista_cores = [
    0.8, 0.8, 0.8, 
    0.8, 0.8, 0.8, 
    0.8, 0.8, 0.8,
    0.8, 0.8, 0.8, 
    0.8, 0.8, 0.8, 
    0.8, 0.8, 0.8 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 9, data_q8)

    # nuvem 2
    lista_cores = [
    0.9, 0.9, 0.9, 
    0.9, 0.9, 0.9, 
    0.9, 0.9, 0.9,
    0.9, 0.9, 0.9, 
    0.9, 0.9, 0.9, 
    0.9, 0.9, 0.9 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 10, data_q9)


    # cor da lua
    lista_cores = [
    1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0 
    ] 

    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 11, data_q6)


    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 12, data_q6) # estrela 1
    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 13, data_q6) # estrela 2
    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 14, data_q6) # estrela 3
    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 15, data_q6) # estrela 4

    # maçaneta
    configura_VAO_e_VBO_quadrado(color_VBO, lista_cores, 16, data_q6)
    


def create_shader_program():
    """
    Compila os códigos fontes dos shaders e 
    armazena o programa shader compilado 
    na variável global shaderProgram.
    """
    global shaderProgram
    global vertex_shader_codigo
    global fragment_shader_codigo

    # Compilar vertex shader
    vertexShader = gl.glCreateShader(gl.GL_VERTEX_SHADER) # Cria objeto shader do tipo GL_VERTEX_SHADER
    gl.glShaderSource(vertexShader, vertex_shader_codigo) # Associa o código fonte ao objeto
    gl.glCompileShader(vertexShader) # Compila o shader

    status = gl.glGetShaderiv(vertexShader, gl.GL_COMPILE_STATUS) # verifica se houve erro na compilação
    if status == gl.GL_FALSE:
        raise RuntimeError("Falha na compilação do shader:" + gl.glGetShaderInfoLog(vertexShader).decode('utf-8'))

    # Compilar fragment shader
    fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragmentShader, fragment_shader_codigo)
    gl.glCompileShader(fragmentShader)

    status = gl.glGetShaderiv(fragmentShader, gl.GL_COMPILE_STATUS)
    if status == gl.GL_FALSE:
        raise RuntimeError("Falha na compilação do shader:" + gl.glGetShaderInfoLog(fragmentShader).decode('utf-8'))

    shaderProgram = gl.glCreateProgram()

    gl.glAttachShader(shaderProgram, vertexShader)
    gl.glAttachShader(shaderProgram, fragmentShader)
    
    gl.glLinkProgram(shaderProgram)
    status = gl.glGetProgramiv(shaderProgram, gl.GL_LINK_STATUS)
    
    if status == gl.GL_FALSE:
        raise RuntimeError("Falha na etapa de link do shader:" + gl.glGetShaderInfoLog(shaderProgram).decode('utf-8'))

    gl.glDeleteShader(vertexShader)  
    gl.glDeleteShader(fragmentShader) 


def display(noturno = False):
    gl.glClearColor(0.1, 0, 0.6, 1.0) if noturno else gl.glClearColor(0.1, 0.7, 0.9, 1.0) 
    # Limpa os buffers de cor e profundidade para cada iteração de renderização
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glEnable(gl.GL_DEPTH_TEST) # Habilita a verificação de profundidade
    gl.glDepthFunc(gl.GL_LESS) # Tipo de comparação para verificar profundidade

    gl.glUseProgram(shaderProgram)

    # Transformação da Matriz View (camera)
    view = glm.mat4(1.0) 
    posicao_camera = glm.vec3(0.0, 0.0, 8.0)
    ponto_camera_aponta = glm.vec3(0.0, 0.0, 0.0)
    vetor_direcao = glm.normalize(posicao_camera - ponto_camera_aponta) 
    vetor_y_mundo = (-1.0)*glm.vec3(0.0, 1.0, 0.0)
    vetor_direita = glm.normalize(glm.cross(vetor_y_mundo, vetor_direcao)) 
    vetor_cima = glm.normalize(glm.cross(vetor_direita, vetor_direcao)) 
    
    # A função LookAt retorna uma matriz do tipo view a partir dos vetores passados por parametro.
    view = glm.lookAt(posicao_camera, ponto_camera_aponta, vetor_cima)
    view_loc = gl.glGetUniformLocation(shaderProgram, "view_matrix") # Localização da variável no shader
    gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view)) # Copia dados para variável no shader

    # Transformacao da Matriz Model 
    # Posicao dos objetos em relação a origem global do mundo
    posicao = glm.vec3(0.0, 0.0, 0.0)
    model = glm.mat4(1.0)
    model = glm.translate(model, posicao)
    model_loc = gl.glGetUniformLocation(shaderProgram, "model_matrix") # Localização da variável no shader
    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model)) # Copia dados para variável no shader



    # Triangulo 1
    display_triangulo(0, -0.65, 0.75, 0.0, 2.5, 2.5, 2.5)

    # parte triangular do telhado
    display_triangulo(1, 2.7, 0.75, 0.0, 2.5, 2.5, 2.5)

    # Quadrado 1
    display_quadrado(2, -2.0, -1.5, -0.1, 3, 3, 3)
 
    # Quadrado 2
    display_quadrado(3, 1.3, -1.5, -0.1, 4.5, 3, 3)
    
    # parte retangular do telhado
    display_quadrado(4, 0.2, 0.28, -0.1, 4, 3, 3)

    # janela 1
    display_quadrado(5, 0.5, -0.8, -0.1, 1, 1, 1)

    # janela 2
    display_quadrado(5, 2.25, -0.8, -0.1, 1, 1, 1)

    # porta 1
    display_quadrado(6, -1.8, -1.69, -0.1, 1, 2, 1)

    # gramado
    display_quadrado(8, -1, -4, -4, 20, 4, 0)

    if noturno == False:
        # sol
        display_quadrado(7, 3, 3, 0, -0.5, .5, 0)

        # nuvem 1
        display_quadrado(9, -0.5, 2.5, 0, 1.5, 0.5, 0)

        # nuvem 2
        display_quadrado(10, 0.2, 2.3, 1, 1.5, 0.5, 0)
    else:
        # lua 
        display_quadrado(11, 3, 3, 0, -0.5, .5, 0)

        
        display_quadrado(12, -0.5, 2.5, 0, -0.1, .1, 0) # estrela 1
        display_quadrado(12, 0.15, 1.8, 0, -0.1, .1, 0) # estrela 1
        display_quadrado(12, -0.27, 1.7, 0, -0.1, .1, 0) # estrela 1
        display_quadrado(12, 10, 1.6, 0, -0.1, .1, 0) # estrela 1

        display_quadrado(13, 0.2, 2.3, 0, -0.1, .1, 0) # estrela 2
        display_quadrado(13, 5, 2.3, 0, -0.1, .1, 0) # estrela 2
        display_quadrado(13, -6, 2.3, 0, -0.1, .1, 0) # estrela 2
        display_quadrado(13, 16, 2.3, 0, -0.1, .1, 0) # estrela 2

        display_quadrado(14, 0.3, 2.7, 0, -0.1, .1, 0) # estrela 3
        display_quadrado(14, -14, 2.7, 0, -0.1, .1, 0) # estrela 3
        display_quadrado(14, -5, 2.7, 0, -0.1, .1, 0) # estrela 3
        display_quadrado(14, -4, 2.7, 0, -0.1, .1, 0) # estrela 3
        display_quadrado(14, -3, 2.7, 0, -0.1, .1, 0) # estrela 3

    # maçaneta
    display_quadrado(16, -1.4, -1.5, 0, -0.1, .1, 0) # estrela 3
    


    gl.glUseProgram(0) # Desvincula o Shader Program

    glut.glutSwapBuffers()


def reshape(width,height):

    gl.glViewport(0, 0, width, height)


def keyboard( key, x, y ):

    print("TECLA PRESSIONADA: {}".format(key))

    if key == b'n': # ESC
        display(True)

    if key == b'd': # ESC
        display()
        

def main_opengl():
    # Cria contexto OpenGL e configura janela
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    largura = 400
    altura = 400
    glut.glutInitWindowSize(largura, altura)
    glut.glutCreateWindow('Av1 - Casa 2d')

    # Inicialização
    # leitura dos vértices dos objetos
    create_shader_program()
    t1 = create_data_triangulo_1()
    t2 = create_data_triangulo_2()
    q1 = create_quad()
    q2 = create_quad2()
    q3 = create_telhado()
    q4 = create_janela()
    q5 = create_porta()
    q6 = create_janela()
    q7 = create_quad()
    q8 = create_quad()
    q9 = create_quad()

    create_buffers(data_t1=t1, data_t2=t2, data_q1=q1, data_q2=q2, data_q3=q3, data_q4 =q4,
        data_q5=q5, data_q6=q6, data_q7=q7, data_q8=q8, data_q9=q9)
    
    # Precisa chamar glUseProgram toda vez antes de acessar variável uniform no shader
    gl.glUseProgram(shaderProgram)

    # Matriz de projeção
    # Cria o "volume de visão" para mapear do "View Space" para o "Clip Space" (NDC)
    # Só precisa ser calculada uma vez, por isso não está dentro da função display
    dist_near = 0.1
    dist_far = 100.0
    projection = glm.mat4(1.0) # Cria matriz identidade 
    projection = glm.perspective(glm.radians(45.0), float(largura)/float(altura), dist_near, dist_far)
    projec_loc = gl.glGetUniformLocation(shaderProgram, "proj_matrix")
    gl.glUniformMatrix4fv(projec_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))

    # Chama funcoes Callback
    glut.glutReshapeFunc(reshape)
    glut.glutDisplayFunc(display)
    glut.glutKeyboardFunc(keyboard)

    
    print("Fornecedor do Driver: {}".format(gl.glGetString(gl.GL_VENDOR).decode()))
    print("Hardware Video: {}".format(gl.glGetString(gl.GL_RENDERER).decode()))
    print("Versao do OpenGL: {}".format(gl.glGetString(gl.GL_VERSION).decode()))

    glut.glutMainLoop()


if __name__ == '__main__':
    main_opengl()
