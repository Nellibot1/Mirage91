import pygame
import random
from PIL import Image

# ─── BCI ADDITION: these three imports are needed for the socket listener ─────
import socket
import threading
import queue
# ─────────────────────────────────────────────────────────────────────────────

# Initialize Pygame
pygame.init()

# Screen Setup
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 300
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BCI Spiel")

# Starting Physics Constants
BASE_GRAVITY = 0.13
BASE_JUMP = -6
FPS = 120
GROUND_Y = SCREEN_HEIGHT - 20

# Load Obstacle Images
bird_img = pygame.image.load('a.jpg').convert_alpha()
bird_img = pygame.transform.scale(bird_img, (100, 40))

cactus_img = pygame.image.load('b.jpg').convert_alpha()
cactus_img = pygame.transform.scale(cactus_img, (50, 80))


# ═══════════════════════════════════════════════════════════════════════════════
# ─── BCI ADDITION: entire BCIListener class is new ────────────────────────────
# Listens for UDP messages sent by main.py (the online BCI pipeline).
# Runs in a background thread so the game loop never freezes waiting for data.
# main.py sends short strings like "DUCK", "ACTION", or "REST" over UDP.
# ═══════════════════════════════════════════════════════════════════════════════

class BCIListener:

    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = port
        self.command_queue = queue.Queue()
        self._running = False

    def start(self):
        self._running = True
        t = threading.Thread(target=self._listen, daemon=True)
        t.start()
        print(f"BCI listener started on {self.host}:{self.port}")

    def stop(self):
        self._running = False

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        while self._running:
            try:
                data, _ = sock.recvfrom(64)
                command = data.decode("utf-8").strip()
                self.command_queue.put(command)
            except socket.timeout:
                continue
        sock.close()

    def get_latest(self):
        """Returns the most recent BCI command, or None if nothing new."""
        command = None
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                break
        return command
# ─── END BCI ADDITION: BCIListener ────────────────────────────────────────────


# ─── GIF LOADER ──────────────────────────────────────────────────────────────

def load_gif(filename):
    pil_image = Image.open(filename)
    frames = []
    try:
        while True:
            frame = pil_image.convert('RGBA')
            pygame_surface = pygame.image.fromstring(
                frame.tobytes(), frame.size, frame.mode
            ).convert_alpha()
            pygame_surface = pygame.transform.scale(pygame_surface, (60, 60))
            frames.append(pygame_surface)
            pil_image.seek(pil_image.tell() + 1)
    except EOFError:
        pass
    return frames


# ─── PLAYER ──────────────────────────────────────────────────────────────────

class Player:
    def __init__(self, frames):
        self.frames = frames
        self.frame_index = 0
        self.image = self.frames[0]
        self.rect = self.image.get_rect(midbottom=(100, GROUND_Y))
        self.velocity = 0
        self.is_ducking = False

    def handle_input(self, jump_val, bci_command):
        """
        ORIGINAL: handle_input(self, jump_val) — keyboard only.
        BCI ADDITION: added bci_command parameter.
            bci_command is the latest string from BCIListener:
            "ACTION" → jump, "DUCK" → crouch, "REST" or None → do nothing.
        Keyboard controls still work as a fallback for testing without the BCI.
        """
        keys = pygame.key.get_pressed()

        # ── Ducking ──────────────────────────────────────────────────
        # BCI ADDITION: bci_crouch triggers on CLENCH command
        bci_crouch = bci_command in {"DUCK", "CLENCH"}
        key_crouch = keys[pygame.K_DOWN] and self.rect.bottom >= GROUND_Y
        self.is_ducking = (bci_crouch or key_crouch)  # BCI ADDITION: was just key_crouch

        # ── Jumping ──────────────────────────────────────────────────
        # BCI ADDITION: bci_jump triggers on BLINK command
        bci_jump = bci_command in {"ACTION", "BLINK"}
        key_jump = (keys[pygame.K_UP] or keys[pygame.K_SPACE])
        if (bci_jump or key_jump) and self.rect.bottom >= GROUND_Y:  # BCI ADDITION: added bci_jump
            self.velocity = jump_val

    def update(self, grav_val):
        self.velocity += grav_val
        self.rect.y += self.velocity
        if self.rect.bottom > GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.velocity = 0

        self.frame_index += 0.2
        if self.frame_index >= len(self.frames):
            self.frame_index = 0

        current_frame = self.frames[int(self.frame_index)]
        if self.is_ducking:
            self.image = pygame.transform.scale(current_frame, (60, 30))
            self.rect = self.image.get_rect(midbottom=(100, GROUND_Y))
        else:
            self.image = current_frame
            self.rect = self.image.get_rect(midbottom=(100, self.rect.bottom))


# ─── OBSTACLE ─────────────────────────────────────────────────────────────────

class Obstacle:
    def __init__(self, type, speed):
        self.type = type
        if self.type == 'air':
            self.image = bird_img
            self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH + 50, GROUND_Y - 50))
        else:
            self.image = cactus_img
            self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH + 50, GROUND_Y + 10))
        self.speed = speed

    def update(self):
        self.rect.x -= self.speed

    def draw(self):
        screen.blit(self.image, self.rect)


# ─── UI HELPERS ───────────────────────────────────────────────────────────────

def show_message(text, size, y_offset=0):
    font = pygame.font.SysFont('Arial', size, bold=True)
    surface = font.render(text, True, (50, 50, 50))
    rect = surface.get_rect(center=(SCREEN_WIDTH // 2, (SCREEN_HEIGHT // 2) + y_offset))
    screen.blit(surface, rect)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    clock = pygame.time.Clock()
    gif_frames = load_gif('aa.gif')
    font = pygame.font.SysFont('Arial', 24)
    font_small = pygame.font.SysFont('Arial', 18)

    # ─── BCI ADDITION: start the background listener before the game loop ────
    bci = BCIListener(host="127.0.0.1", port=5005)
    bci.start()
    # ─────────────────────────────────────────────────────────────────────────

    game_active = True
    player = Player(gif_frames)
    obstacles = []
    spawn_timer = 0
    score = 0
    last_bci_command = "REST"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if not game_active and event.key == pygame.K_r:
                    game_active = True
                    player = Player(gif_frames)
                    obstacles = []
                    spawn_timer = 0
                    score = 0

        # ─── BCI ADDITION: poll for latest command each frame ────────────────
        bci_command = bci.get_latest()
        if bci_command:
            last_bci_command = bci_command
        # ─────────────────────────────────────────────────────────────────────

        if game_active:
            # Dynamic difficulty
            difficulty_level = int(score // 80)
            current_speed   = 3 + (difficulty_level * 0.4)
            current_gravity = BASE_GRAVITY + (difficulty_level * 0.01)
            current_jump    = BASE_JUMP - (difficulty_level * 0.1)
            current_min     = max(120, 200 - (difficulty_level * 2))
            current_max     = max(200, 250 - (difficulty_level * 3))

            # Pass BCI command into player
            player.handle_input(current_jump, bci_command)  # BCI ADDITION: added bci_command arg
            player.update(current_gravity)

            spawn_timer += 1
            if spawn_timer > random.randint(current_min, current_max):
                obstacles.append(Obstacle(random.choice(['ground', 'air']), current_speed))
                spawn_timer = 0

            for obstacle in obstacles[:]:
                obstacle.update()
                if player.rect.colliderect(obstacle.rect):
                    game_active = False
                if obstacle.rect.right < 0:
                    obstacles.remove(obstacle)

            score += 0.1

            # Drawing
            screen.fill((255, 255, 255))
            screen.blit(player.image, player.rect)
            for obs in obstacles:
                obs.draw()

            pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

            score_text = font.render(
                f'Score: {int(score)}  LVL: {difficulty_level + 1}', True, (0, 0, 0)
            )
            screen.blit(score_text, (10, 10))

            # ─── BCI ADDITION: live status label showing current BCI command ─
            # Colour coded: grey=REST, green=ACTION/BLINK, orange=DUCK/CLENCH
            cmd_color = {
                "REST":   (150, 150, 150),
                "ACTION": (50,  180, 80),
                "BLINK":  (50,  180, 80),
                "DUCK":   (220, 140, 0),
                "CLENCH": (220, 140, 0),
            }.get(last_bci_command, (0, 0, 0))
            bci_text = font_small.render(f'BCI: {last_bci_command}', True, cmd_color)
            screen.blit(bci_text, (10, 40))
            # ─────────────────────────────────────────────────────────────────

        else:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill((255, 255, 255))
            screen.blit(overlay, (0, 0))
            show_message("GAME OVER", 64, -40)
            show_message(f"Final Score: {int(score)}", 32, 20)
            show_message("Press 'R' to Restart", 24, 75)

        pygame.display.flip()
        clock.tick(FPS)

    bci.stop()  # BCI ADDITION: cleanly shut down the background listener thread
    pygame.quit()


if __name__ == "__main__":
    main()
