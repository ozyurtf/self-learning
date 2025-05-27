from matplotlib.pylab import *
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import torch
import torch.nn as nn
from torchviz import make_dot
from tqdm import tqdm
from datetime import datetime
from random import seed, uniform
import wandb
import scipy.stats as stats
import matplotlib.style as style
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Training/Testing Code of Truck Backer Upper")

parser.add_argument("--train_emulator", type=str, default = "False", required=False)
parser.add_argument("--train_controller", type=str, default = "False", required=False)
parser.add_argument("--num_test_trajectories", type=int, default = 10, required=False)

parser.add_argument("--env_x_range", type=int, nargs=2, default = (0, 40), required=False)
parser.add_argument("--env_y_range", type=int, nargs=2, default = (-15, 15), required=False)

parser.add_argument("--train_x_cab_range", type=int, nargs=2, default = (5, 35), required=False)    
parser.add_argument("--train_y_cab_range_abs", type=int, nargs=2, default = (2, 7), required=False)
parser.add_argument("--train_cab_angle_range_abs", type=int, nargs=2, default = (10, 90), required=False)
parser.add_argument("--train_cab_trailer_angle_diff_range_abs", type=int, nargs=2, default = (10, 45), required=False)
parser.add_argument("--train_num_lessons", type=int, default = 10, required=False)

parser.add_argument("--test_x_cab_range", type=int, nargs=2, default = (5, 35), required=False)
parser.add_argument("--test_y_cab_range", type=int, nargs=2, default = (-7, 7), required=False)
parser.add_argument("--test_cab_angle_range", type=int, nargs=2, default = (-90, 90), required=False)
parser.add_argument("--test_cab_trailer_angle_diff_range", type=int, nargs=2, default = (-45, 45), required=False)
parser.add_argument("--test_lesson", type=int, default = 10, required=False)

parser.add_argument("--display_trajectories", type=str, default = "False", required=False)
parser.add_argument("--truck_speed", type=float, default = -0.1, required=False)
parser.add_argument("--wandb_log", type=str, default = "False", required = False)
parser.add_argument("--save_computational_graph", type=str, default = "False", required=False)

args = parser.parse_args()

train_emulator_flag = args.train_emulator=="True"
train_controller_flag = args.train_controller=="True"
num_test_trajectories = args.num_test_trajectories

train_x_cab_range = args.train_x_cab_range
train_y_cab_range_abs = args.train_y_cab_range_abs
train_cab_angle_range_abs = args.train_cab_angle_range_abs
train_cab_trailer_angle_diff_range_abs = args.train_cab_trailer_angle_diff_range_abs
train_num_lessons = args.train_num_lessons

test_x_cab_range = args.test_x_cab_range
test_y_cab_range = args.test_y_cab_range
test_cab_angle_range = args.test_cab_angle_range
test_cab_trailer_angle_diff_range = args.test_cab_trailer_angle_diff_range

env_x_range = args.env_x_range
env_y_range = args.env_y_range
display_trajectories = args.display_trajectories=="True"

test_lesson = args.test_lesson
truck_speed = args.truck_speed
wandb_log = args.wandb_log=="True"
save_computational_graph = args.save_computational_graph=="True"

current_time = datetime.now().strftime("%Y-%m-%d_%I-%M%p")
π = pi
style.use(['dark_background', 'bmh'])

def create_train_configs(num_lessons):
    configs = {}
    first_x_cab, final_x_cab = train_x_cab_range
    first_y_cab, final_y_cab = train_y_cab_range_abs
    first_cab_angle, final_cab_angle = train_cab_angle_range_abs
    first_cab_trailer_angle_diff, final_cab_trailer_angle_diff = train_cab_trailer_angle_diff_range_abs
    x_lower = first_x_cab

    for i in range(1, num_lessons):
        x_upper = first_x_cab + (final_x_cab - first_x_cab) * (i - 1) / (num_lessons - 1)
        y_upper = first_y_cab + (final_y_cab - first_y_cab) * (i - 1) / (num_lessons - 1)        
        θ0_upper = first_cab_angle + (final_cab_angle - first_cab_angle) * (i - 1) / (num_lessons - 1)
        Δθ_upper = first_cab_trailer_angle_diff + (final_cab_trailer_angle_diff - first_cab_trailer_angle_diff) * (i - 1) / (num_lessons - 1)
        
        configs[i] = {"x_range": (x_lower, x_upper),
                      "y_range": (-y_upper, y_upper),
                      "θ0_range": (-θ0_upper, θ0_upper),
                      "Δθ_range": (-Δθ_upper, Δθ_upper)}
        x_lower = x_upper
        
    configs[num_lessons] = {"x_range": (first_x_cab, x_upper),
                            "y_range": (-y_upper, y_upper),
                            "θ0_range": (-θ0_upper, θ0_upper),
                            "Δθ_range": (-Δθ_upper, Δθ_upper)}
            
    return configs

train_configs = create_train_configs(train_num_lessons)
test_config = {"x_range": test_x_cab_range,
               "y_range": test_y_cab_range,
               "θ0_range": test_cab_angle_range,
               "Δθ_range": test_cab_trailer_angle_diff_range}

class Truck:
    def __init__(self, lesson, display=False):

        self.W = 1 
        self.L = 1 * self.W 
        self.d = 4 * self.L 
        self.s = truck_speed
        self.display = display
        self.lesson = lesson
        
        self.box = [0, env_x_range[1], env_y_range[0], env_y_range[1]]
        if self.display:
            self.f = figure(figsize=(10, 6), num='The Truck Backer-Upper', facecolor='none')
            self.ax = self.f.add_axes([0.01, 0.01, 0.98, 0.98], facecolor='black')
            self.patches = list()
                
            self.ax.axis('equal')
            b = self.box
            self.ax.axis([b[0] - 1, b[1], b[2], b[3]])
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.ax.axhline(); self.ax.axvline()
            
            plt.ion()  
            plt.pause(0.001) 
        
        self.trailer_trajectory = []
        self.cab_trajectory = []
        
    def reset(self, ϕ=0, train_test = "train", test_seed = 1):
        self.trailer_trajectory.clear()
        self.cab_trajectory.clear()
        
        self.ϕ = ϕ 
        if train_test == "train":
            config = train_configs.get(self.lesson)
        else: 
            config = test_config
        
        if config is None: 
            raise ValueError(f"No configuration found")       

        if train_test == "test": 
            seed(test_seed)
            
        self.x = uniform(*config["x_range"])
        self.y = uniform(*config["y_range"])                    
        self.θ0 = deg2rad(uniform(*config["θ0_range"]))
        self.θ1 = deg2rad(uniform(*config["Δθ_range"])) + self.θ0
    
        if not self.valid():
            self.reset(ϕ)
        
        if self.display: 
            self.draw()        
    
    def step(self, ϕ=0, dt=1):
        
        if self.is_jackknifed():
            print('The truck is jackknifed!')
            return
        
        if self.is_offscreen():
            print('The car or trailer is off screen')
            return
        
        self.ϕ = ϕ
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        self.x += s * cos(θ0) * dt
        self.y += s * sin(θ0) * dt
        self.θ0 += s / L * tan(ϕ) * dt
        self.θ1 += s / d * sin(θ0 - θ1) * dt   

        trailer_pos = (self._trailer_xy()[0], self._trailer_xy()[1])
        cab_pos = (self.x, self.y)

        self.trailer_trajectory.append(trailer_pos) 
        self.cab_trajectory.append(cab_pos)
                        
        return (self.x, self.y, self.θ0, *self._trailer_xy(), self.θ1)
    
    def state(self):
        return (self.x, self.y, self.θ0, *self._trailer_xy(), self.θ1)
    
    def _get_atributes(self):
        return (
            self.x, self.y, self.W, self.L, self.d, self.s,
            self.θ0, self.θ1, self.ϕ
        )
    
    def _trailer_xy(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        return x - d * cos(θ1), y - d * sin(θ1)
        
    def is_jackknifed(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()   
        diff_deg = rad2deg(θ0) - rad2deg(θ1) 
        abs_diff_deg = abs(diff_deg)
        return min(abs_diff_deg,  abs(abs_diff_deg - 360)) > 90
    
    def is_offscreen(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        x1, y1 = x + 1.5 * L * cos(θ0), y + 1.5 * L * sin(θ0)
        x2, y2 = self._trailer_xy()
        
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
        plt.pause(0.001)            
            
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

        car = Rectangle((x1, y1 - W / 2),   
                        L,                  
                        W,                  
                        color='C2',        
                        alpha=1,
                        transform=(matplotlib.transforms.Affine2D().rotate_deg_around(x1, y1, rad2deg(θ0)) +
                                   ax.transData))

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
        trailer = Rectangle((x, y),   
                            d,        
                            W,        
                            color='C0', 
                            alpha=1,
                            transform = (matplotlib.transforms.Affine2D().rotate_deg_around(x, y + W / 2, rad2deg(θ1)) + 
                                         ax.transData))

        ax.add_patch(trailer)
        
        self.patches += [trailer]

    def _draw_trajectories(self, test_seed): 
                        
        x_trailer_trajectory = [point[0] for point in self.trailer_trajectory]
        y_trailer_trajectory = [point[1] for point in self.trailer_trajectory]
        
        x_cab_trajectory = [point[0] for point in self.cab_trajectory]
        y_cab_trajectory = [point[1] for point in self.cab_trajectory]
        
        style.use('seaborn-v0_8-whitegrid') 
        
        plt.figure(figsize=(7.5, 3), dpi=100) 
        
        trailer_color = '#1f77b4'  
        cab_color = '#ff7f0e'      
        
        plt.plot(x_trailer_trajectory, y_trailer_trajectory, 
                 color=trailer_color, linestyle='-', linewidth=1.5, alpha=0.8)
        
        plt.plot(x_cab_trajectory, y_cab_trajectory, 
                 color=cab_color, linestyle='-', linewidth=1.5, alpha=0.8)
        
        plt.scatter(x_trailer_trajectory, y_trailer_trajectory, 
                    color=trailer_color, marker='.', s=15, alpha=0.6)
        
        plt.scatter(x_cab_trajectory, y_cab_trajectory, 
                    color=cab_color, marker='.', s=15, alpha=0.6)
        
        plt.scatter(x_trailer_trajectory[0], y_trailer_trajectory[0], 
                    marker='o', color=trailer_color, s=60, zorder=10, 
                    label='Trailer Start Position')
        
        plt.scatter(x_cab_trajectory[0], y_cab_trajectory[0], 
                    marker='o', color=cab_color, s=60, zorder=10,
                    label='Cab Start Position')
                
        plt.scatter(x_trailer_trajectory[-1], y_trailer_trajectory[-1], 
                   marker='x', color=trailer_color, s=60, zorder=10,
                   label='Trailer End Position')
        
        plt.scatter(x_cab_trajectory[-1], y_cab_trajectory[-1], 
                   marker='x', color=cab_color, s=60, zorder=10,
                   label='Cab End Position')
        
        
        plt.plot([x_trailer_trajectory[0], x_cab_trajectory[0]], 
                 [y_trailer_trajectory[0], y_cab_trajectory[0]], 
                 'k--', linewidth=1.5) 

        plt.plot([x_trailer_trajectory[-1], x_cab_trajectory[-1]], 
                 [y_trailer_trajectory[-1], y_cab_trajectory[-1]], 
                 'k--', linewidth=1.5)
        
        plt.xlim(env_x_range[0], env_x_range[1])
        plt.ylim(env_y_range[0], env_y_range[1])

        rectangle_red = patches.Rectangle((train_x_cab_range[0], -train_y_cab_range_abs[1]), train_x_cab_range[1]-train_x_cab_range[0], train_y_cab_range_abs[1] * 2,
                                            facecolor='red',           
                                            edgecolor='darkred',       
                                            alpha=0.3,                 
                                            linewidth=2)
        
        plt.gca().add_patch(rectangle_red)     
        
        red_x0 = train_x_cab_range[0]
        red_y0 = -train_y_cab_range_abs[1]
        red_width = train_x_cab_range[1] - red_x0
        red_height = train_y_cab_range_abs[1] * 2

        plt.text(red_x0 + red_width - 0.2, 
                 red_y0 + red_height - 0.2,
                 'Training Region',
                 color='white',
                 ha='right',
                 va='top',
                 fontsize=5,
                 fontweight='bold')

                
        rectangle_green = patches.Rectangle((env_x_range[0], env_y_range[0]), env_x_range[1] - env_x_range[0], env_y_range[1] - env_y_range[0],
                                            facecolor='green',           
                                            edgecolor='darkgreen',       
                                            alpha=0.3,                 
                                            linewidth=2)
        
        plt.gca().add_patch(rectangle_green)     
        
        green_x0 = env_x_range[0]
        green_y0 = env_y_range[0]
        green_width = env_x_range[1] - green_x0
        green_height = env_y_range[1] - green_y0

        plt.text(green_x0 + green_width - 0.2,  
                 green_y0 + green_height - 0.2,
                 'Testing Region',
                 color='white',
                 ha='right',
                 va='top',
                 fontsize=5,
                 fontweight='bold')
        
        plt.scatter(0, 0, marker='x', color="darkgray", s=60, zorder=10, label = "Target") 
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        for spine in plt.gca().spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cccccc')
        
        plt.tick_params(axis='both', which='major', labelsize=8, pad=4, colors='#555555')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.78)
        
        directory = f'trajectories/lesson-{self.lesson}'
        
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        fig = plt.gcf() 
        fig.patch.set_facecolor('white')      
        plt.savefig(f'{directory}/trajectory-{test_seed}.png', dpi=300, facecolor='white', bbox_inches='tight')  
        if display_trajectories==False:
            plt.close()    
        
    def update_state(self, state): 
        self.ϕ, self.x, self.y, self.θ0, self.θ1 = state.tolist()
        
def generate_random_deg(mean = 0, std = 35, lower_bound = -70, upper_bound = 70):     
    a = (lower_bound - mean) / std
    b = (upper_bound - mean) / std
    samples = stats.truncnorm.rvs(a, b, loc = mean, scale = std, size = 1)
    sample = samples[0]
    return sample

def initialize_emulator(): 
    emulator = nn.Sequential(
        nn.Linear(5, 100),
        nn.GELU(),
        nn.Linear(100,100),
        nn.GELU(),        
        nn.Linear(100, 4)
    )
    torch.save(emulator, 'models/emulators/emulator_lesson_0.pth')
    return emulator


def initialize_controller():
    controller = nn.Sequential( 
        nn.Linear(5, 100),
        nn.GELU(),
        nn.Linear(100, 100),
        nn.GELU(),        
        nn.Linear(100, 1),
    )
    torch.save(controller, 'models/controllers/controller_lesson_0.pth')
    return controller

criterion_emulator = nn.MSELoss()  

def criterion_controller(ϕ_state):
    ϕ, x, y, θ0, θ1 = ϕ_state 
    x_tr, y_tr = x - 4 * torch.cos(θ1), y - 4 * torch.sin(θ1)
    x_tr_relu = nn.functional.relu(x_tr)
    abs_diff_rad = torch.abs(θ0 - θ1) 
    abs_diff_deg = torch.rad2deg(abs_diff_rad)
    angle_diff_deg_relu = nn.functional.relu(torch.min(abs_diff_deg, abs(abs_diff_deg - 360)) - 90)
    min_θ1 = torch.min(torch.abs(θ1), torch.abs(torch.abs(θ1) - deg2rad(360)))
    return (x_tr_relu**2 + y_tr**2 + min_θ1**2 + angle_diff_deg_relu**2) / 4

def train_emulator(emulator, 
                   episodes, 
                   learning_rate, 
                   lesson, 
                   wandb_log = wandb_log):
    
    if wandb_log:
        wandb.init(project='emulator-training', save_code = True, name=f'lesson_{lesson}_run_{current_time}')
        
    inputs = list()
    outputs = list()
    truck = Truck(lesson)
    for episode in tqdm(range(episodes)):
        truck.reset()
        while truck.valid():
            x, y, θ0, _, _, θ1 = truck.state()
            random_deg = generate_random_deg()
            ϕ = deg2rad(random_deg) 
            inputs.append((ϕ, x, y, θ0, θ1))
            x_next, y_next, θ0_next, _, _, θ1_next = truck.step(ϕ)
            outputs.append((x_next, y_next, θ0_next, θ1_next))
            
    tensor_inputs = torch.Tensor(inputs)
    tensor_outputs = torch.Tensor(outputs)
    
    test_size = int(len(tensor_inputs) * 0.8)
    
    train_inputs = tensor_inputs[:test_size]
    train_outputs = tensor_outputs[:test_size]
    
    test_inputs = tensor_inputs[test_size:]
    test_outputs = tensor_outputs[test_size:]
    
    print("Train Size:", len(train_inputs))
    print("Test Size:", len(test_inputs))
    
    optimizer = torch.optim.Adam(emulator.parameters(), lr=learning_rate)
    
    global_step = 0
    for i in torch.randperm(len(train_inputs)):
        ϕ_state = train_inputs[ i]
        
        next_state_prediction = emulator(ϕ_state)
        next_state = train_outputs[i]
        
        optimizer.zero_grad()
        loss = criterion_emulator(next_state_prediction, next_state)
        loss.backward()
        
        if wandb_log:
            wandb.log({'train_loss': loss.item(),
                       'gradients': {name: param.grad.norm().item() for name, param in emulator.named_parameters() if param.grad is not None}}, step=global_step)
        
        optimizer.step()
        global_step += 1

    with torch.no_grad():
        total_loss = 0
        for j in range(len(test_inputs)):
            ϕ_state = test_inputs[j]
            next_state = test_outputs[j]
            next_state_prediction = emulator(ϕ_state)
            loss = criterion_emulator(next_state_prediction, next_state)
            total_loss += loss.item()
            if wandb_log: 
                wandb.log({'test_loss': loss.item()}, step = global_step)
            global_step += 1

    test_size = len(test_inputs)
    avg_test_loss = total_loss / test_size
    
    print()
    print(f'Test loss: {avg_test_loss:.10f}')
    
    torch.save(emulator, f'models/emulators/emulator_lesson_{lesson}.pth')
    
    if wandb_log:
        wandb.finish()
    
    return emulator

def train_controller(lesson, 
                     controller, 
                     epochs, 
                     max_steps,
                     learning_rate,
                     wandb_log = wandb_log,
                     save_computational_graph = save_computational_graph):
      
    if wandb_log: 
        wandb.init(project='controller-training', save_code = True, name=f'lesson_{lesson}_run_{current_time}')
      
    emulator = torch.load(f'./models/emulators/emulator_lesson_{lesson}.pth', weights_only=False)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    truck = Truck(lesson, display=False)
    
    for i in tqdm(range(epochs)):
        truck.reset()
        x, y, θ0, _, _, θ1 = truck.state()
        ϕ = truck.ϕ
        ϕ_state = torch.tensor([ϕ, x, y, θ0, θ1], dtype=torch.float32)
        step = 0
        
        while step <= max_steps and truck.valid():
            ϕ_prediction = controller(ϕ_state)
            next_state_prediction = emulator(ϕ_state)
            ϕ_state = torch.cat((ϕ_prediction, next_state_prediction))
            truck.update_state(ϕ_state)
            step += 1
 
        optimizer.zero_grad()
        loss = criterion_controller(ϕ_state)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters = controller.parameters(),               
                                       max_norm = 5, 
                                       error_if_nonfinite = True)        
        
        if wandb_log:
            wandb.log({'loss': loss.item(),
                       'gradients': {name: param.grad.norm().item() for name, param in controller.named_parameters() if param.grad is not None}}, step=i)

        
        if save_computational_graph and i == 0 and lesson == 1: 
            dot = make_dot(loss, params=dict(controller.named_parameters()))
            dot.format = 'png'
            dot.render('computational-graphs/controller_computational_graph')
        
        optimizer.step()
        
        if i % 100 == 0:
            torch.save(controller, 'models/controllers/controller_lesson_{}.pth'.format(lesson))
            loss_value = loss.item()
            print(f'{loss_value:.10f}')
    
    if wandb_log:
        wandb.finish()
            
    return controller

if train_emulator_flag:
    
    emulators_dir = 'models/emulators'
    
    if os.path.exists(emulators_dir):
        shutil.rmtree(emulators_dir)
    os.makedirs(emulators_dir)
    
    emulator = initialize_emulator()

    for lesson in range(1, train_num_lessons + 1):
        print(" Lesson {}:".format(lesson))
        emulator = train_emulator(lesson = lesson,
                                emulator = emulator,
                                episodes = 10_000,
                                learning_rate = 0.00001)
        print()
        
    train_controller_flag = True
        
if train_controller_flag:        
    controllers_dir = 'models/controllers'
    
    if os.path.exists(controllers_dir):
        shutil.rmtree(controllers_dir)
    os.makedirs(controllers_dir)

    controller = initialize_controller()

    for lesson in range(1, train_num_lessons + 1): 
        print(" Lesson {}:".format(lesson))
        controller = train_controller(lesson = lesson, 
                                      controller = controller,
                                      epochs = 3000,
                                      max_steps = 400,
                                      learning_rate = 0.0001)
        print()

test_controller = torch.load('models/controllers/controller_lesson_{}.pth'.format(test_lesson), weights_only = False)
truck = Truck(lesson = test_lesson, display = True)

num_jackknifes = 0
for test_seed in range(1, num_test_trajectories + 1):
    with torch.no_grad():
        truck.reset(train_test = "test", test_seed = test_seed)    
        ϕ = torch.tensor([truck.ϕ])
        i = 0
        while truck.valid():
            x, y, θ0, _, _, θ1 = truck.state()
            state = torch.tensor([x, y, θ0, θ1], dtype = torch.float32) 
            ϕ_state = torch.cat((ϕ, state))
            next_ϕ = test_controller(ϕ_state) 
            truck.step(ϕ.item())
            truck.draw()
            ϕ = next_ϕ
            i += 1
        truck._draw_trajectories(test_seed)
        x, y, θ0, trailer_x, trailer_y, θ1 = truck.state()  
        num_jackknifes += truck.is_jackknifed()
        print(f"Number of Steps: {i}")
        print(f"Is Jackknifed ? {truck.is_jackknifed()}")
        print(f"Trailer x: {trailer_x:.3f}, Trailer y: {trailer_y:.3f}")
        print()
print(f"Number of Jackknifes: {num_jackknifes}")