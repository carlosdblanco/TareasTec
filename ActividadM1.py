%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import datetime
import numpy as np
import pandas as pd
import random

from mesa import Agent, Model 

from mesa.space import MultiGrid

from mesa.time import RandomActivation

from mesa.datacollection import DataCollector

plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128


def obtener_habitacion(model):
    habitacion = np.zeros((model.grid.width, model.grid.height))
    for celda in model.grid.coord_iter():
        contenido_celda, x, y = celda
        for contenido in contenido_celda:
            if isinstance(contenido, Aspiradora):
                habitacion[x][y] = 2
            else:
                habitacion[x][y] = contenido.estado
    return habitacion

class Aspiradora(Agent): # Aspiradora
    def __init__(self,unique_id,model):
        super().__init__(unique_id, model)
        self.estado = 3 # Estado de aspiradora
        
    def move(self): # Movimiento 
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def limpiarPiso(self): # Limpieza
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for mate in cellmates:
                if mate.estado == 1:
                    mate.estado = 0
                    self.model.restante -= 1

    def step(self):
        self.limpiarPiso()
        self.move()

class Suciedad(Agent): #La suciedad
    def __init__(self,unique_id,model):
        super().__init__(unique_id, model)
        self.estado = 1 #Estado 1 esta sucio

    def step(self):
        if self.estado == 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)


class LimpiezaModel(Model):
    def __init__(self, N, M, width, height):
        self.aspiradoras = N
        self.suciedades = M
        self.restante = M
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        for i in range(self.aspiradoras):
            a = Aspiradora(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        for i in range(self.suciedades):
            a = Suciedad(i+M, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width) #Random
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector( 
            model_reporters={"Grid": obtener_habitacion})
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

     
GRID_SIZE = 10 # Definimos el tamaño

MAX_TIME = 0.01 # Tiempo maximo

CELDAS_SUCIAS = 50 # Celdas sucias

ASPIRADORAS = 5 #Cantidad de aspiradoras

start_time = time.time()
step_num = 0
restantes = CELDAS_SUCIAS
modelo = LimpiezaModel(ASPIRADORAS, CELDAS_SUCIAS, GRID_SIZE, GRID_SIZE)
while time.time()-start_time <= MAX_TIME:
    modelo.step()
    step_num += 1
    if modelo.restante < 1:
        break
    
final_time = time.time()

all_grid = modelo.datacollector.get_model_vars_dataframe()

fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate(i):
    patch.set_data(all_grid.iloc[i][0])
    
anim = animation.FuncAnimation(fig, animate, frames=step_num)

print('Porcentaje inicial de suciedad:', str((CELDAS_SUCIAS / (GRID_SIZE * GRID_SIZE)) * 100), '%') # Celdas sucias y limpias
print('Porcentaje final de suciedad:', str((modelo.restante / (GRID_SIZE * GRID_SIZE)) * 100), '%')

print('Tiempo de ejecución:', str(datetime.timedelta(seconds=(final_time - start_time)))) #Imprimimos el tiempo de ejecucion

print('Cantidad de movimientos:', step_num) #Movimientos
