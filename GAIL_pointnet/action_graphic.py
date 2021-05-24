import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import pygame
# import OPENGL

BLACK = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255,0,0)


class action_screen():
	def __init__(self, box_size = 120):
		pygame.init()
		self.screen = pygame.display.set_mode((5*box_size,box_size))
		sysfont = pygame.font.get_default_font()
		self.font = pygame.font.SysFont('freesansbold.ttf', 72)
		self.quit = False
		self.box_size = box_size
		self.quit = False
		self.screen.fill(WHITE)

	def display(self, action):
		
		# while not self.quit:
		# try:
		pygame.event.get()
		print(action)
		for i,p in enumerate(action):
			a = self.font.render(str(p), True, BLACK)
			color = (255,int(255*(1-p)),int(255*(1-p)))
			pygame.draw.rect(self.screen, color, (i*self.box_size,0,self.box_size, self.box_size))
			self.screen.blit(a, (i*self.box_size,0))
			
		
		# pygame.display.update()
		# except KeyboardInterrupt:
		# 	break

	def finish(self):
		self.quit = True


ascreen = action_screen()
ascreen.display([0.1,0,0.5])
ascreen.finish()