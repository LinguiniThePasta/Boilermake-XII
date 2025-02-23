import pygame
import pyperclip

pygame.font.init()
FONT = pygame.font.Font(None, 16)

BLACK = (0,0,0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

class InputBox:
    def __init__(self, x, y, w, h, label, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = GRAY
        self.text = text
        self.txt_surface = FONT.render(text, True, BLACK)
        self.label_surface = FONT.render(label, True, BLACK)
        self.active = False
        self.label = label

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the user clicked on the input box
            self.active = self.rect.collidepoint(event.pos)
            self.color = BLACK if self.active else GRAY

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                print(self.text)  # Print input text
                self.text = ''  # Clear input box
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_v and (event.mod & pygame.KMOD_CTRL or event.mod & pygame.KMOD_META):
                # Handle Paste
                pasted_text = pyperclip.paste()
                self.text += pasted_text  # Append pasted text
            else:
                self.text += event.unicode # Handle normal typing

            self.txt_surface = FONT.render(self.text, True, BLACK)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 2)
        text_y = self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2
        screen.blit(self.txt_surface, (self.rect.x + 5, text_y))
        screen.blit(self.label_surface, (self.rect.x, self.rect.y - 20))

    def get_value(self):
        return self.text


class Button:
    def __init__(self, x, y, w, h, text='', action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = (0, 255, 0)
        self.text = text
        self.txt_surface = FONT.render(text, True, BLACK)
        self.active = False
        self.action = action

    def handle_event(self, event):
        # Check if the button was clicked
        if event.type == pygame.MOUSEBUTTONDOWN:
            print("HI!")
            if self.rect.collidepoint(event.pos):
                if self.action is not None:
                    self.action() 
                return True
        return False
        

    def draw(self, screen):
        # Fill the button with the color
        pygame.draw.rect(screen, self.color, self.rect)  # Solid green fill
        text_y = self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2
        screen.blit(self.txt_surface, (self.rect.x + 5, text_y))
