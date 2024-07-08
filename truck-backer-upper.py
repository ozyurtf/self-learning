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
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
import os 
import shutil
# from torch.utils.tensorboard import SummaryWriter

π = pi

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

style.use(['dark_background', 'bmh'])

class Truck:
    def __init__(self, lesson, display=False):

        self.W = 1  # car and trailer width, for drawing only
        self.L = 1 * self.W  # car length
        self.d = 4 * self.L  # d_1
        self.s = -0.1  # speed
        self.display = display
        self.lesson = lesson

        
        self.box = [0, 40, -10, 10]
        if self.display:
            self.f = figure(figsize=(6, 3), num='The Truck Backer-Upper', facecolor='none')
            self.ax = self.f.add_axes([0.01, 0.01, 0.98, 0.98], facecolor='black')
            self.patches = list()
            
            self.ax.axis('equal')
            b = self.box
            self.ax.axis([b[0] - 1, b[1], b[2], b[3]])
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.ax.axhline(); self.ax.axvline()
    
    def reset(self, ϕ=0):
        self.ϕ = ϕ  
           
        if self.lesson == 1: 
            random_deg_θ0 = uniform(0, 10)
            random_deg_θ1 = uniform(-10, 10)
            self.θ0 = deg2rad(random_deg_θ0)   
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0
            self.x = uniform(10, 15)
            self.y = uniform(-2, 2)
          
        elif self.lesson == 2: 
            random_deg_θ0 = uniform(0, 30) 
            random_deg_θ1 = uniform(-20, 20)
            self.θ0 = deg2rad(random_deg_θ0) 
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0
            self.x = uniform(15, 20)
            self.y = uniform(-4, 4)     
                        
        elif self.lesson == 3: 
            random_deg_θ0 = uniform(0, 40) 
            random_deg_θ1 = uniform(-30, 30)
            self.θ0 = deg2rad(random_deg_θ0) 
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0
            self.x = uniform(20, 25)
            self.y = uniform(-5, 5)                 
                        
        elif self.lesson == 4: 
            random_deg_θ0 = uniform(0, 60)
            random_deg_θ1 = uniform(-40, 40)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(20, 25)
            self.y = uniform(-6, 6) 
            
        elif self.lesson == 5: 
            random_deg_θ0 = uniform(0, 70)
            random_deg_θ1 = uniform(-40, 40)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(20, 25)
            self.y = uniform(-6, 6)   
            
        elif self.lesson == 6: 
            random_deg_θ0 = uniform(0, 70)
            random_deg_θ1 = uniform(-40, 40)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(25, 30)
            self.y = uniform(-6, 6)               
            
        elif self.lesson == 7: 
            random_deg_θ0 = uniform(0, 80)
            random_deg_θ1 = uniform(-40, 40)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(25, 30)
            self.y = uniform(-6, 6)                  
            
        elif self.lesson == 8: 
            random_deg_θ0 = uniform(0, 90)
            random_deg_θ1 = uniform(-45, 45)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(30, 35)
            self.y = uniform(-7, 7)   
            
        elif self.lesson == 9: 
            random_deg_θ0 = uniform(0, 90)
            random_deg_θ1 = uniform(-45, 45)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(10, 35)
            self.y = uniform(-7, 7)  
            
        elif self.lesson == 10: 
            random_deg_θ0 = uniform(0, 120)
            random_deg_θ1 = uniform(-45, 45)
            self.θ0 = deg2rad(random_deg_θ0)            
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0  
            self.x = uniform(10, 35)
            self.y = uniform(-7, 7)         
                    
        elif self.lesson == 11: 
            random_deg_θ0 = uniform(0, 360)   
            random_deg_θ1 = uniform(-45, 45)  
            self.θ0 = deg2rad(random_deg_θ0) 
            self.θ1 = deg2rad(random_deg_θ1) + self.θ0    
            self.x = uniform(10, 35)
            self.y = uniform(-7, 7)         
                                        
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
        bar = Line2D((x, x1), (y, y1), lw=5, color='C2', alpha=1)
        ax.add_line(bar)

        car = Rectangle(
            (x1, y1 - W / 2), L, W, 0, color='C2', alpha=1, transform=
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
            (x, y), d, W, 0, color='C0', alpha=1, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x, y + W/2, rad2deg(θ1)) +
            ax.transData
        )
        ax.add_patch(trailer)
        
        self.patches += [trailer]


def generate_random_deg(mean, std, scale_lower, scale_upper): 
    random_num = np.random.normal(loc = mean, scale = std)
    random_num_scaled = np.clip(random_num, scale_lower, scale_upper)
    return random_num_scaled

def initialize_emulator(): 
    emulator = nn.Sequential(
      nn.Linear(7, 100),
      nn.GELU(),
      nn.Linear(100, 100),
      nn.GELU(), 
      nn.Linear(100, 6)  
    )

    torch.nn.init.normal_(emulator[0].weight, 
                          mean = 0.0, 
                          std = 0.01)

    torch.nn.init.normal_(emulator[2].weight, 
                          mean = 0.0, 
                          std = 0.01) 

    torch.nn.init.normal_(emulator[4].weight, 
                          mean = 0.0, 
                          std = 0.01)

    torch.save(emulator, 'models/emulators/emulator_lesson_0.pth')
    return emulator


def initialize_controller():
    controller = nn.Sequential( 
        nn.Linear(7, 50),
        nn.GELU(),
        nn.Linear(50, 1)
    )

    torch.nn.init.normal_(controller[0].weight,
                          mean = 0, 
                          std = 0.1)

    torch.nn.init.normal_(controller[2].weight, 
                          mean = 0, 
                          std = 0.1)
          
    torch.save(controller, 'models/controllers/controller_lesson_0.pth')

    return controller 



def train_emulator(emulator,  
                   episodes, 
                   learning_rate,
                   lesson):
  
    inputs = list()
    outputs = list()
    truck = Truck(lesson)

    for episode in tqdm(range(episodes)):
        truck.reset()
        while truck.valid():
            initial_state = truck.state()
            random_deg = generate_random_deg(mean = 0,
                                             std = 35,
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
    
    optimiser = torch.optim.Adam(emulator.parameters(), lr=learning_rate)    

    criterion = nn.MSELoss()  

    for i in torch.randperm(len(train_inputs)):
        ϕ_state = train_inputs[i]
        next_state_prediction = emulator(ϕ_state)

        next_state = train_outputs[i]
        loss = criterion(next_state_prediction, next_state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters = emulator.parameters(),               
                                       max_norm = 5, 
                                       error_if_nonfinite = True)            
        optimiser.step()
        optimiser.zero_grad()
    
      
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

    torch.save(emulator, 'models/emulators/emulator_lesson_{}.pth'.format(lesson))  

    return emulator


# !rm -rf ./runs/

def criterion(x, y, θ1, θ0, step):
    angle_diff_rad = torch.abs(θ1 - θ0)
    angle_diff_deg = torch.rad2deg(angle_diff_rad)
    angle_diff_relu = nn.functional.relu((angle_diff_deg - 30)/30)
    x_relu = nn.functional.relu(x)
    min_θ1 = torch.min(torch.abs(θ1), torch.abs(torch.abs(θ1) - torch.deg2rad(torch.tensor(360.0))))
    part1 = x_relu + torch.abs(y) + min_θ1
    part2 = (x_relu**2 + y**2 + min_θ1**2 + angle_diff_relu**2 + step * 0.01)
    
    return -torch.log(1 / (part1 * part2))

def train_controller(lesson, 
                     emulator,
                     controller, 
                     epochs, 
                     max_steps,
                     learning_rate = 0.0001):
    
    optimiser = torch.optim.Adam(controller.parameters(), lr = learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = 1000, gamma = 0.9)
    
    loss_values = []
        
    truck = Truck(lesson, display = False)
    L = truck.L 
    b = truck.box 
    # writer = SummaryWriter()
    
    for i in tqdm(range(epochs)):
        step = torch.tensor(0.0, requires_grad = True)
        truck.reset()
        random_deg = generate_random_deg(mean = 0,
                                         std = 70,
                                         scale_lower = -70,
                                         scale_upper = 70)
        ϕ = deg2rad(random_deg)
        
        initial_state = truck.state()
        state = torch.cat((torch.tensor([ϕ], requires_grad = True, dtype=torch.float32), 
                           torch.tensor(initial_state, requires_grad = True, dtype=torch.float32))) 
        
        x, y, θ0, x2, y2, θ1 = state[1:]
        x1, y1 = x + 1.5 * L * torch.cos(θ0), y + 1.5 * L * torch.sin(θ0)
        is_jacknifed = torch.rad2deg(abs(θ0 - θ1)) > 90
        is_offscreen = not (b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and
                            b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3])   

        while step <= max_steps and is_jacknifed == False and is_offscreen == False:    
            ϕ_pred = controller(state)
            next_state_prediction = emulator(state)
            state = torch.cat((ϕ_pred, next_state_prediction))
            x, y, θ0, x2, y2, θ1 = state[1:]
            x1, y1 = x + 1.5 * L * torch.cos(θ0), y + 1.5 * L * torch.sin(θ0)
            is_jacknifed = torch.rad2deg(abs(θ0 - θ1)) > 90
            is_offscreen = not (b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and
                                b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3]) 
            step = step + 1
                    
        loss = criterion(x = x2, 
                         y = y2,
                         θ0 = θ0,  
                         θ1 = θ1,
                         step = step)     
        loss.backward() 
        
        # for name, param in controller.named_parameters():
        #    if param.grad is not None:
        #        writer.add_histogram(f'lesson{lesson}-layer{name}.grad', param.grad, i)
            
        torch.nn.utils.clip_grad_norm_(parameters = controller.parameters(),               
                                       max_norm = 1, 
                                       error_if_nonfinite = True)

        optimiser.step()
        optimiser.zero_grad() 
        scheduler.step()
        
        if i % 100 == 0:    
            torch.save(controller, 'models/controllers/controller_lesson_{}.pth'.format(lesson))
            loss_value = loss.item()
            loss_values.append(loss_value)
            print(f'{loss_value:.10f}')  
            # writer.add_scalar(f'lesson{lesson}-loss', loss_value, i) 
    
    # writser.close()
    return controller      


emulators_directory = 'models/controllers'

if os.path.exists(emulators_directory):
    shutil.rmtree(emulators_directory)

os.makedirs(emulators_directory, exist_ok=True)

emulator = initialize_emulator()

for lssn in range(1,11): 
    print("Lesson {}:".format(lssn))
    train_emulator(lesson = lssn,
                   emulator = emulator, 
                   episodes = 10_000,
                   learning_rate = 0.00001)
    print()


controllers_directory = 'models/controllers'

if os.path.exists(controllers_directory):
    shutil.rmtree(controllers_directory)

os.makedirs(controllers_directory, exist_ok=True)


controller_0 = initialize_controller()


controller_1 = train_controller(lesson = 1, 
                                emulator = emulator,
                                controller = controller_0,                                  
                                epochs = 3000,
                                max_steps = 100)



controller_2 = train_controller(lesson = 2, 
                                emulator = emulator,
                                controller = controller_1,                                                                    
                                epochs = 3000, 
                                max_steps = 150)


controller_3 = train_controller(lesson = 3, 
                                emulator = emulator,
                                controller = controller_2,                                                                            
                                epochs = 3000, 
                                max_steps = 200)


controller_4 = train_controller(lesson = 4,
                                emulator = emulator,
                                controller = controller_3,                                          
                                epochs = 3000, 
                                max_steps = 225)


controller_5 = train_controller(lesson = 5, 
                                emulator = emulator,
                                controller = controller_4,  
                                epochs = 3000, 
                                max_steps = 250)

controller_6 = train_controller(lesson = 6, 
                                emulator = emulator,
                                controller = controller_5,  
                                epochs = 3000, 
                                max_steps = 275)


controller_7 = train_controller(lesson = 7, 
                                emulator = emulator,
                                controller = controller_6,
                                epochs = 3000, 
                                max_steps = 300)

controller_8 = train_controller(lesson = 8, 
                                emulator = emulator,
                                controller = controller_7,
                                epochs = 3000,                                 
                                max_steps = 300)

controller_9 = train_controller(lesson = 9, 
                                emulator = emulator,
                                controller = controller_8,
                                epochs = 3000,                                 
                                max_steps = 350)


controller_10 = train_controller(lesson = 10, 
                                 emulator = emulator,
                                 controller = controller_9,
                                 epochs = 3000,                                 
                                 max_steps = 350)

controller_11 = train_controller(lesson = 11, 
                                 emulator = emulator,
                                 controller = controller_10,
                                 epochs = 3000,                                 
                                 max_steps = 350)


lesson = 10
test_controller = torch.load('models/controllers/controller_lesson_{}.pth'.format(lesson))
truck = Truck(lesson = lesson, display=True)
truck.reset()

with torch.no_grad():
    truck.reset()
    truck.ϕ = 0
    i = 0
    while truck.valid():
        t1 = torch.tensor([truck.ϕ],dtype=torch.float32)
        state = truck.state()      
        t2 = torch.Tensor(state)
        state = torch.cat((t1,t2))
        ϕ = test_controller(state)
        truck.step(ϕ.item())
        truck.draw()
        i += 1
    print("Number of Steps: {}".format(i))