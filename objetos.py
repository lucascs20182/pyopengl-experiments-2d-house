import sys
import glm # Versão em Python da OpenGL Mathematics (GLM)
import OpenGL.GL as gl # Funcoes da API OpenGL
import OpenGL.GLUT as glut # Criacao de janelas acesso ao teclado
import numpy as np
import ctypes


# Variáveis globais
shaderProgram = None
VBO = [None, None, None] # Vertex Buffer Object
VAO = [None, None, None] # Vertex Array Object

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


def create_data_triangulo_1():
    
    # Vertices do triangulo
    v1 = glm.vec3(-0.9, -0.3, 0.0) # esquerda abaixo
    v2 = glm.vec3(0.1, -0.3, 0.0) # meio abaixo
    v3 = glm.vec3(-0.35, 0.3, 0.0) # esquerda acima 
    tri = glm.mat3(v1, v2, v3) 
    return tri


def create_data_triangulo_2():
    
    # Vertices do triangulo
    v1 = glm.vec3(0.0, -0.5, 0.0) # 
    v2 = glm.vec3(0.9, -0.5, 0.0) # 
    v3 = glm.vec3(0.45, 0.5, 0.0) # 
    tri = glm.mat3(v1, v2, v3) 
    return tri

def create_quad():

    lista_vertices = [
        -0.5, -0.5, -0.5, # Face 1 | Triangulo 1 | Vertice 1
        0.5, -0.5, -0.5, # Face 1 | Triangulo 1 | Vertice 2
        0.5,  0.5, -0.5, # Face 1 | Triangulo 1 | Vertice 3
        0.5,  0.5, -0.5, # Face 1 | Triangulo 2 | Vertice 1
        -0.5,  0.5, -0.5, # Face 1 | Triangulo 2 | Vertice 2
        -0.5, -0.5, -0.5 # Face 1 | Triangulo 2 | Vertice 3
        ]

    vertices = np.array(lista_vertices, dtype=np.float32)
    
    return vertices

def create_buffers(data_t1, data_t2, data_q1):

    global VAO
    global VBO 
    color_VBO = [None, None, None] 

    VAO[0] = gl.glGenVertexArrays(1) 
    VAO[1] = gl.glGenVertexArrays(1) 
    VAO[2] = gl.glGenVertexArrays(1) 

    VBO[0] = gl.glGenBuffers(1) 
    VBO[1] = gl.glGenBuffers(1)
    VBO[2] = gl.glGenBuffers(1)

    color_VBO[0] = gl.glGenBuffers(1)
    color_VBO[1] = gl.glGenBuffers(1)
    color_VBO[2] = gl.glGenBuffers(1)

    # Triangulo 1
    gl.glBindVertexArray(VAO[0])
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO[0]) # Efetua o bind do VBO
    gl.glBufferData(target=gl.GL_ARRAY_BUFFER, size= glm.sizeof(data_t1), data=glm.value_ptr(data_t1), usage=gl.GL_STATIC_DRAW)
    local = gl.glGetAttribLocation(shaderProgram, 'vPos')
    vertexDim = 3 # quantidade de elementos do vetor declarado no shader
    stride = 0 # Espaço em bytes até o próximo valor. E.g. próximo x, quando for posição (X | Y | Z | X | Y | ...)
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local, vertexDim, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    # Copia os dados para o color_VBO
    lista_cores = [
    1.0, 0.0, 0.0, # Vertice 1
    1.0, 0.0, 0.0, # Vertice 2
    1.0, 0.0, 0.0 # Vertice 3
    ] 
    colors_to_buffer = np.array(lista_cores, dtype=np.float32)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_VBO[0]) # Efetua o bind do VBO
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

    # Triangulo 2
    gl.glBindVertexArray(VAO[1])
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO[1]) # Efetua o bind do VBO
    gl.glBufferData(target=gl.GL_ARRAY_BUFFER, size= glm.sizeof(data_t2), data=glm.value_ptr(data_t2), usage=gl.GL_STATIC_DRAW)
    local = gl.glGetAttribLocation(shaderProgram, 'vPos')
    vertexDim = 3 # quantidade de elementos do vetor declarado no shader
    stride = 0 # Espaço em bytes até o próximo valor. E.g. próximo x, quando for posição (X | Y | Z | X | Y | ...)
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local, vertexDim, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    # Copia os dados para o color_VBO
    lista_cores = [
    0.0, 3.0, 2.0, # Vertice 1
    0.0, 3.0, 2.0, # Vertice 2
    0.0, 3.0, 2.0 # Vertice 3
    ] 
    colors_to_buffer = np.array(lista_cores, dtype=np.float32)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_VBO[1]) # Efetua o bind do VBO
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

    # Quadrado 1
    gl.glBindVertexArray(VAO[2])
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO[2]) # Efetua o bind do VBO
    gl.glBufferData(gl.GL_ARRAY_BUFFER, (ctypes.c_float*len(data_q1))(*data_q1), gl.GL_STATIC_DRAW)
    local = gl.glGetAttribLocation(shaderProgram, 'vPos')
    vertexDim = 3 # quantidade de elementos do vetor declarado no shader
    stride = 0 # Espaço em bytes até o próximo valor. E.g. próximo x, quando for posição (X | Y | Z | X | Y | ...)
    offset = None # Onde os dados iniciam no Vertex Buffer
    # Descreve a forma de organização dos dados dentro do último buffer (VBO) vinculado (glBindBuffer)
    gl.glVertexAttribPointer(local, vertexDim, gl.GL_FLOAT, gl.GL_FALSE, stride, offset) 
    gl.glEnableVertexAttribArray(local) # Associa e habilita os dados do Vertex Buffer (VBO) no Array

    # Copia os dados para o color_VBO
    lista_cores = [
    1.0, 0.0, 1.0, # Vertice 1
    1.0, 0.0, 1.0, # Vertice 2
    1.0, 0.0, 1.0,# Vertice 3
    1.0, 0.0, 1.0, # Vertice 1
    1.0, 0.0, 1.0, # Vertice 2
    1.0, 0.0, 1.0 # Vertice 3
    ] 
    colors_to_buffer = np.array(lista_cores, dtype=np.float32)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_VBO[2]) # Efetua o bind do VBO
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


def display():

    # gl.glClearColor(0.5, 0.5, 0.5, 1.0)
    # gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    gl.glClearColor(0.5, 0.5, 0.5, 1.0)
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
    gl.glBindVertexArray(VAO[0])

    # Cria matrizes de transformação
    matriz_transformacao = glm.identity(glm.mat4) # Cria matriz identidade
    # Quanto eu quero transladar em cada eixo
    vetor_translacao = glm.vec3(-0.65, 0.75, 0.0)
    matriz_transformacao = glm.translate(matriz_transformacao, vetor_translacao)

    # Aumentar o tamanho do quadrado (escala)
    escalar = 2.5
    # Aplica um mesmo valor de escala em todos os eixos
    vetor_escala = glm.vec3(escalar, escalar, escalar)
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

    
    # Triangulo 2
    gl.glBindVertexArray(VAO[1])

    # Cria matrizes de transformação
    matriz_transformacao = glm.identity(glm.mat4) # Cria matriz identidade
    # Quanto eu quero transladar em cada eixo
    vetor_translacao = glm.vec3(1.5, 1.0, 0.0)
    matriz_transformacao = glm.translate(matriz_transformacao, vetor_translacao)

    # Localizacao da variavel Uniform matriz transformacao no vertex shader
    transformacao_loc = gl.glGetUniformLocation(shaderProgram, "transformacao")
    # Copia os dados da matriz de transformacao para matriz Uniform dentro do vertex shader
    gl.glUniformMatrix4fv(transformacao_loc, 1, gl.GL_FALSE, glm.value_ptr(matriz_transformacao))

    quant_vertices = 3 
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, quant_vertices)
    gl.glBindVertexArray(0) # Desvincula o VAO

    # Quadrado 1
    gl.glBindVertexArray(VAO[2])

    # Cria matrizes de transformação
    matriz_transformacao = glm.identity(glm.mat4) # Cria matriz identidade
    # Quanto eu quero transladar em cada eixo
    vetor_translacao = glm.vec3(-2.0, -1.5, -0.1)
    matriz_transformacao = glm.translate(matriz_transformacao, vetor_translacao)

    # Aumentar o tamanho do quadrado (escala)
    escalar = 3.0
    # Aplica um mesmo valor de escala em todos os eixos
    vetor_escala = glm.vec3(escalar, escalar, escalar)
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

    gl.glUseProgram(0) # Desvincula o Shader Program

    glut.glutSwapBuffers()


def reshape(width,height):

    gl.glViewport(0, 0, width, height)


def keyboard( key, x, y ):

    print("TECLA PRESSIONADA: {}".format(key))

    if key == b'\x1b': # ESC
        sys.exit( )  

def main_opengl():
    print(" ==== main_opengl ====")

    # Cria contexto OpenGL e configura janela
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
    largura = 400
    altura = 400
    glut.glutInitWindowSize(largura, altura)
    glut.glutCreateWindow('Desenhando mais de um objeto')

    # Inicialização
    create_shader_program()
    t1 = create_data_triangulo_1()
    t2 = create_data_triangulo_2()
    q1 = create_quad()
    create_buffers(data_t1=t1, data_t2=t2, data_q1=q1)
    
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
    print("\nOBJETOS")

    main_opengl()