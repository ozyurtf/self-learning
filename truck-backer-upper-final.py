from matplotlib.pylab import *
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
from random import uniform
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'
import wandb

π = pi

import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

style.use(['dark_background', 'bmh'])
get_ipython().run_line_magic('matplotlib', 'notebook')

class Truck:
    def __init__(self, iteration, display=False):

        self.W = 1  # car and trailer width, for drawing only
        self.L = 1 * self.W  # car length
        self.d = 4 * self.L  # d_1
        self.s = -0.1  # speed
        self.display = display
        self.iteration = iteration

        
        self.box = [0, 40, -10, 10]
        if self.display:
            self.f = figure(figsize=(6, 3), num='The truck backer-upper', facecolor='none')
            self.ax = self.f.add_axes([0.01, 0.01, 0.98, 0.98], facecolor='black')
            self.patches = list()
            
            self.ax.axis('equal')
            b = self.box
            self.ax.axis([b[0] - 1, b[1], b[2], b[3]])
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.ax.axhline(); self.ax.axvline()
    
    def reset(self, ϕ=0):
        self.ϕ = ϕ  
           
        if self.iteration == 1: 
          random_deg_θ0 = uniform(0, 10)
          random_deg_θ1 = uniform(-5, 5)
          self.θ0 = deg2rad(random_deg_θ0)   
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0
          self.x = uniform(10, 10)
          self.y = uniform(-3, 3)
        
        elif self.iteration == 2: 
          random_deg_θ0 = uniform(0, 20) 
          random_deg_θ1 = uniform(-5, 5)
          self.θ0 = deg2rad(random_deg_θ0)
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0
          self.x = uniform(10, 15)
          self.y = uniform(-3, 3)        
        
        elif self.iteration == 3: 
          random_deg_θ0 = uniform(0, 30)
          random_deg_θ1 = uniform(-10, 10)
          self.θ0 = deg2rad(random_deg_θ0)
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0
          self.x = uniform(15, 20)
          self.y = uniform(-3, 3)   
          
        elif self.iteration == 4: 
          random_deg_θ0 = uniform(0, 45) 
          random_deg_θ1 = uniform(-20, 20)
          self.θ0 = deg2rad(random_deg_θ0) 
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0
          self.x = uniform(15, 20)
          self.y = uniform(-5, 5)        
          
        elif self.iteration == 5: 
          random_deg_θ0 = uniform(0, 60)  
          random_deg_θ1 = uniform(-20, 20)
          self.θ0 = deg2rad(random_deg_θ0) 
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0
          self.x = uniform(20, 30)
          self.y = uniform(-8, 8)  
          
        elif self.iteration == 6: 
          random_deg_θ0 = uniform(0,75)  
          random_deg_θ1 = uniform(-30, 30)
          self.θ0 = deg2rad(random_deg_θ0) 
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0
          self.x = uniform(30, 35)
          self.y = uniform(-8, 8)            
          
        elif self.iteration == 7: 
          random_deg_θ0 = uniform(0, 90)
          random_deg_θ1 = uniform(-45, 45)
          self.θ0 = deg2rad(random_deg_θ0)            # 0 <= ϑ₀ < π/2
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0  # -π/4 <= ϑ₁ - ϑ₀ < π/4
          self.x = (random() * .75 + 0.25) * self.box[1]
          self.y = (random() - 0.5) * (self.box[3] - self.box[2])   
          
        elif self.iteration == 8: 
          random_deg_θ0 = uniform(0, 120)
          random_deg_θ1 = uniform(-45, 45)
          self.θ0 = deg2rad(random_deg_θ0)            
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
          self.x = (random() * .75 + 0.25) * self.box[1]
          self.y = (random() - 0.5) * (self.box[3] - self.box[2])   
          
        elif self.iteration == 9: 
          random_deg_θ0 = uniform(0, 180)
          random_deg_θ1 = uniform(-45, 45)
          self.θ0 = deg2rad(random_deg_θ0)            
          self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
          self.x = (random() * .75 + 0.25) * self.box[1]
          self.y = (random() - 0.5) * (self.box[3] - self.box[2])              
          
        elif self.iteration == 10: 
          self.θ0 = random() * 2 * π  # 0 <= ϑ₀ < 2π
          self.θ1 = (random() - 0.5) * π / 2 + self.θ0  # -π/4 <= ϑ₁ - ϑ₀ < π/4
          self.x = (random() * .75 + 0.25) * self.box[1]
          self.y = (random() - 0.5) * (self.box[3] - self.box[2])        
        
        # If poorly initialise, then re-initialise
        if not self.valid():
            self.reset(ϕ)
        
        # Draw, if display is True
        if self.display: 
          self.draw()
    
    def step(self, ϕ=0, dt=1):
        
        # Check for illegal conditions
        if self.is_jackknifed():
            print('The truck is jackknifed!')
            return
        
        if self.is_offscreen():
            print('The car or trailer is off screen')
            return
        
        self.ϕ = ϕ
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        # Perform state update
        self.x += s * cos(θ0) * dt
        self.y += s * sin(θ0) * dt
        self.θ0 += s / L * tan(ϕ) * dt
        self.θ1 += s / d * sin(θ0 - θ1) * dt
        
        return (self.x, self.y, self.θ0, *self._traler_xy(), self.θ1)
    
    def state(self):
        return (self.x, self.y, self.θ0, *self._traler_xy(), self.θ1)
    
    def _get_atributes(self):
        return (
            self.x, self.y, self.W, self.L, self.d, self.s,
            self.θ0, self.θ1, self.ϕ
        )
    
    def _traler_xy(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        return x - d * cos(θ1), y - d * sin(θ1)
        
    def is_jackknifed(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        angle_diff_rad = abs(θ0 - θ1)
        angle_diff_deg = rad2deg(angle_diff_rad)
        return angle_diff_deg > 90
    
    def is_offscreen(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        x1, y1 = x + 1.5 * L * cos(θ0), y + 1.5 * L * sin(θ0)
        x2, y2 = self._traler_xy()
        
        b = self.box
        return not (
            b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and
            b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3]
        )
        
    def valid(self):
        return not self.is_jackknifed() and not self.is_offscreen()
        
    def draw(self):
        if not self.display: return
        if self.patches: self.clear()
        self._draw_car()
        self._draw_trailer()
        self.f.canvas.draw()
            
    def clear(self):
        for p in self.patches:
            p.remove()
        self.patches = list()
        
    def _draw_car(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()

        ax = self.ax
        
        x1, y1 = x + L / 2 * cos(θ0), y + L / 2 * sin(θ0)
        bar = Line2D((x, x1), (y, y1), lw=5, color='C2', alpha=0.8)
        ax.add_line(bar)

        car = Rectangle(
            (x1, y1 - W / 2), L, W, 0, color='C2', alpha=0.8, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x1, y1, rad2deg(θ0)) +
            ax.transData
        )
        ax.add_patch(car)

        x2, y2 = x1 + L / 2 ** 0.5 * cos(θ0 + π / 4), y1 + L / 2 ** 0.5 * sin(θ0 + π / 4)
        left_wheel = Line2D(
            (x2 - L / 4 * cos(θ0 + ϕ), x2 + L / 4 * cos(θ0 + ϕ)),
            (y2 - L / 4 * sin(θ0 + ϕ), y2 + L / 4 * sin(θ0 + ϕ)),
            lw=3, color='C5', alpha=1)
        ax.add_line(left_wheel)

        x3, y3 = x1 + L / 2 ** 0.5 * cos(π / 4 - θ0), y1 - L / 2 ** 0.5 * sin(π / 4 - θ0)
        right_wheel = Line2D(
            (x3 - L / 4 * cos(θ0 + ϕ), x3 + L / 4 * cos(θ0 + ϕ)),
            (y3 - L / 4 * sin(θ0 + ϕ), y3 + L / 4 * sin(θ0 + ϕ)),
            lw=3, color='C5', alpha=1)
        ax.add_line(right_wheel)
        
        self.patches += [car, bar, left_wheel, right_wheel]
        
    def _draw_trailer(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()    
        ax = self.ax
             
        x, y = x - d * cos(θ1), y - d * sin(θ1) - W / 2
        trailer = Rectangle(
            (x, y), d, W, 0, color='C0', alpha=0.8, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x, y + W/2, rad2deg(θ1)) +
            ax.transData
        )
        ax.add_patch(trailer)
        
        self.patches += [trailer]
        
def generate_random_num(mean, std, scale_lower, scale_upper): 
    random_num = np.random.normal(loc = mean, scale = std)
    random_num_scaled = np.clip(random_num, scale_lower, scale_upper)
    return random_num_scaled         

style.use(['dark_background', 'bmh'])

truck = Truck(iteration = 8, display=True)
truck.reset()

def train_emulator(emulator,  
                   episodes, 
                   learning_rate,
                   iteration):
  
  inputs = list()
  outputs = list()
  truck = Truck(iteration)
  truck.reset()

  for episode in tqdm(range(episodes)):
    truck.reset()
    while truck.valid():
      initial_state = truck.state()
      random_deg = generate_random_num(mean = 0,
                                       std = 70,
                                       scale_lower = -70,
                                       scale_upper = 70)
      ϕ = deg2rad(random_deg)
      inputs.append((ϕ, *initial_state))
      outputs.append(truck.step(ϕ))
        
        
  tensor_inputs = torch.Tensor(inputs)
  tensor_outputs = torch.Tensor(outputs)

  test_size = int(len(tensor_inputs) * 0.8)

  train_inputs = tensor_inputs[:test_size]
  train_outputs = tensor_outputs[:test_size]
  test_inputs = tensor_inputs[test_size:]
  test_outputs = tensor_outputs[test_size:]

  print("Train Size:", len(train_inputs))
  print("Test Size:", len(test_inputs)) 
  
  optimiser = torch.optim.SGD(emulator.parameters(), lr=learning_rate, nesterov = True, momentum = 0.9)
  
  criterion = nn.MSELoss()  
  
  for i in torch.randperm(len(train_inputs)):
      ϕ_state = train_inputs[i]
      next_state_prediction = emulator(ϕ_state)

      next_state = train_outputs[i]
      loss = criterion(next_state_prediction, next_state)

      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
      
  total_loss = 0
  with torch.no_grad():
    for idx, ϕ_state in enumerate(test_inputs):
      next_state_prediction = emulator(ϕ_state)
      next_state = test_outputs[idx]
      total_loss += criterion(next_state_prediction, next_state).item()
  
  test_size = len(test_inputs)
  avg_test_loss = total_loss/test_size
 
  print()
  print(f'Test loss: {avg_test_loss:.10f}')  
  
  torch.save(emulator, 'emulator_final_iteration{}.pth'.format(iteration))  
  
  return emulator

def train_controller(emulator, 
                     
                     controller, 
                     
                     epochs, 
                     
                     max_steps,
                     
                     angle_diff_threshold,
                     
                     angle_diff_penalty,
                     
                     momentum,
                     
                     iteration, 
                     
                     clip_grad_norm,
                     
                     learning_rate):
  
  optimiser = torch.optim.SGD(controller.parameters(), 
                              
                              lr = learning_rate, 
                              
                              nesterov = True, 
                              
                              momentum = momentum)
  
  def criterion(x, y, θ1, θ0, step): 
    
      angle_diff_deg = torch.rad2deg(abs(θ1 - θ0))
      
      angle_diff_relu = nn.functional.relu((angle_diff_deg-angle_diff_threshold)/angle_diff_threshold)
      
      return (x**2 + 
              
              y**4 + 
              
              min(θ1**2, (θ1-deg2rad(360))**2, (θ1+deg2rad(360))**2) + 
              
              step*0.001 + 
              
              angle_diff_relu**angle_diff_penalty)
  
  truck = Truck(iteration)
  
  L = truck.L 
  
  b = truck.box 
  
  emulator.requires_grad = False

  for i in tqdm(range(epochs)):
    
      step = torch.tensor(0.0, requires_grad = True)

      truck.reset()

      ϕ = truck.ϕ 

      initial_state = truck.state()

      state = torch.cat((torch.tensor([ϕ], requires_grad = True, dtype=torch.float32), 
                         
                         torch.tensor(initial_state, requires_grad = True, dtype=torch.float32))) 

      x = state[1]
      
      y = state[2]
      
      θ0 = state[3]
      
      x2 = state[4]
      
      y2 = state[5]
      
      θ1 = state[6]
      
      x1, y1 = x + 1.5 * L * torch.cos(θ0), y + 1.5 * L * torch.sin(θ0)
      
      is_jacknifed = torch.rad2deg(abs(θ0 - θ1)) > 90
      
      is_offscreen = not (b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and
                          b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3])   

      while step <= max_steps and is_jacknifed == False and is_offscreen == False:    
        
          ϕ_pred = controller(state)
          
          next_state_prediction = emulator(state)
          
          state = torch.cat((ϕ_pred, next_state_prediction))

          x = state[1]
          
          y = state[2]
          
          θ0 = state[3]
          
          x2 = state[4]
          
          y2 = state[5]
          
          θ1 = state[6]
          
          x1, y1 = x + 1.5 * L * torch.cos(θ0), y + 1.5 * L * torch.sin(θ0)
          
          is_jacknifed = torch.rad2deg(abs(θ0 - θ1)) > 90
          
          is_offscreen = not (b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and
                              
                              b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3]) 

          step = step + 1
          
      optimiser.zero_grad()    

      loss = criterion(x = state[4], 
                       
                       y = state[5], 
                       
                       θ0 = state[3], 
                       
                       θ1 = state[6], 
                       
                       step = step)     

      loss.backward()  

      torch.nn.utils.clip_grad_norm_(parameters = controller.parameters(), 
                                     
                                     max_norm = clip_grad_norm, 
                                     
                                     error_if_nonfinite = True)

      optimiser.step()

      if i % 200 == 0:
        
          torch.save(controller, 'controller_final_iteration{}.pth'.format(iteration))
          
          print(f'{loss.item():.10f}')  
          
  return controller, emulator      

iteration = 0 

emulator = nn.Sequential(
    nn.Linear(7, 100),
    nn.GELU(),
    nn.Linear(100, 6)
)

controller = nn.Sequential( 
    nn.Linear(7, 100),
    nn.GELU(),
    nn.Linear(100, 1) 
)

torch.nn.init.normal_(emulator[0].weight, 
                      mean = 0.0, 
                      std = 0.01)

torch.nn.init.normal_(emulator[2].weight, 
                      mean = 0.0, 
                      std = 0.01)

torch.nn.init.normal_(controller[0].weight, 
                      mean = 0.0, 
                      std = 0.01)

torch.nn.init.normal_(controller[2].weight, 
                      mean = 0.0, 
                      std = 0.01)

torch.save(emulator, 'emulator_final_iteration{}.pth'.format(iteration))
torch.save(controller, 'controller_final_iteration{}.pth'.format(iteration))

iteration = 1

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.0005,
                          iteration = iteration)


controller, emulator = train_controller(emulator = emulator, 
                                        controller = controller, 
                                        angle_diff_threshold = 20,
                                        angle_diff_penalty = 4,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 1,
                                        momentum = 0.75,                                        
                                        iteration = iteration, 
                                        epochs = 10_000, 
                                        max_steps = 75)

iteration = 2

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.0005,
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator, 
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 10,
                                        angle_diff_penalty = 5,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 1,
                                        momentum = 0.75,                                           
                                        epochs = 5000, 
                                        max_steps = 100)

iteration = 3 

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.0005,
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator, 
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 10,                                        
                                        angle_diff_penalty = 5,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 1,
                                        momentum = 0.75,                                           
                                        epochs = 5000, 
                                        max_steps = 150)

iteration = 4

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.0005,                          
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator, 
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 10, 
                                        angle_diff_penalty = 6,                                          
                                        learning_rate = 0.001,     
                                        clip_grad_norm = 0.1,      
                                        momentum = 0.75,           
                                        epochs = 2000, 
                                        max_steps = 150)

iteration = 5

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.00005,                              
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator,
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 5,                                          
                                        angle_diff_penalty = 6,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 0.01,
                                        momentum = 0.75,                                           
                                        epochs = 5000, 
                                        max_steps = 220)

iteration = 6 

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.0005,                              
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator,
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 5,                                          
                                        angle_diff_penalty = 3,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 0.1,
                                        momentum = 0.90,                                           
                                        epochs = 10_000, 
                                        max_steps = 300)

iteration = 7

emulator = train_emulator(emulator = emulator, 
                          episodes = 10_000,
                          learning_rate = 0.0005,                              
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator,
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 5,                                          
                                        angle_diff_penalty = 3,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 0.1,
                                        momentum = 0.90,                                           
                                        epochs = 10_000, 
                                        max_steps = 350)

iteration = 8

emulator = train_emulator(emulator = emulator, 
                          episodes = 20_000,
                          learning_rate = 0.00005,                              
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator,
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 5,                                          
                                        angle_diff_penalty = 6,                                        
                                        learning_rate = 0.01,
                                        clip_grad_norm = 0.01,
                                        momentum = 0.90,                                           
                                        epochs = 10_000, 
                                        max_steps = 350)

iteration = 9

emulator = train_emulator(emulator = emulator, 
                          episodes = 20_000,
                          learning_rate = 0.00005,                              
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator,
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 60,                                          
                                        angle_diff_penalty = 8,                                        
                                        learning_rate = 0.0001,
                                        clip_grad_norm = 0.001,
                                        momentum = 0.90,                                           
                                        epochs = 20_000, 
                                        max_steps = 350)

iteration = 10

emulator = train_emulator(emulator = emulator, 
                          episodes = 20_000,
                          learning_rate = 0.00005,                              
                          iteration = iteration)

controller, emulator = train_controller(emulator = emulator,
                                        controller = controller, 
                                        iteration = iteration, 
                                        angle_diff_threshold = 90,                                          
                                        angle_diff_penalty = 8,                                        
                                        learning_rate = 0.001,
                                        clip_grad_norm = 0.001,
                                        momentum = 0.05,                                           
                                        epochs = 20_000, 
                                        max_steps = 400)

