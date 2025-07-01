import pygame
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import  cosine_similarity

movies=[
    ("3 Idiots","comedy drams education friendship collage life"),
    ("John Wick","Gun friendship action revenge assicination"),
    ("Dil Se Digital", "romance sci-fi virtual love AI futuristic"),
    ("Gully Beat", "rap music struggle ambition slum dreams"),
    ("Shaadi Squad", "romcom wedding chaos family drama love"),
    ("Metro Mein Milan", "city life strangers destiny romance train"),
    ("Mission Ishq", "spy thriller undercover love betrayal action"),
    ("Kho Gayi Khushi", "mystery love memory-loss emotional thriller"),
    ("Pyaar Tera Code", "romantic tech coding competition youth"),
    ("Andher Nagri", "political satire corruption dark humor town"),
    ("Zindagi Restart", "self-discovery drama journey inspiration dreams"),
    ("Chalte Chalte", "journey romance life experience relationships"),
    ("Silent Monsoon", "artistic love emotion deaf mute seasonal"),
    ("Rikshaw Romeo", "romance action dreamer daily-struggle city"),
    ("Desi Detective", "crime mystery investigation small-town twist"),
    ("Dil Ka Dhamaka", "romance comedy marriage celebration family"),
    ("Mission Bollywood", "undercover agent industry corruption drama"),
    ("Thok Ke Bol", "courtroom justice lawyer crime intense drama"),
    ("Safar-E-Ishq", "romantic journey mountains heartbreak healing"),
    ("Laptop Aur Love", "college romance technology misunderstanding connection"),
    ("Pyaar Ka Plot Twist", "romcom family secrets surprise engagement"),
    ("Bollywood Hacker", "cyber thriller AI tech blackmail mystery"),
    ("Magic Ki Shaadi", "romantic comedy magic wedding fantasy"),
    ("Biryani Blues", "food rivalry romance cultural fusion contest"),
    ("Ghanti Baj Gayi", "supernatural comedy ghost temple funny"),
    ("Dosti Returns", "friendship reunion drama past secrets emotions"),
    ("Gully Ki Garmi", "female rapper struggle rebellion slum beat"),
    ("Zindagi Ka Exam", "student pressure family education choices drama"),
    ("Pyaar Wali Politics", "romantic satire election drama college"),
    ("Mission Mangal 2", "space patriotism science women empowerment team"),
    ("Metro Raaz", "thriller secrets city mystery suspense"),
    ("Shaadi.com Se Shaadi Tak", "romcom online match families drama wedding"),
    ("Ishq 4.0", "sci-fi love robot emotion future heartbreak"),
    ("Khoobsurat Kaand", "comedy heist glamour chase madness"),
    ("Code Dil Se", "tech romance app heartbreak hackathon"),
    ("College Ka Canteen", "teen comedy friendship crush music"),
    ("DJ Dhamaka", "party love betrayal remix emotions"),
    ("Silent Safar", "emotional journey mute love scenery"),
    ("Tandoori Dreams", "culinary love street food competition passion"),
    ("Gehra Raaz", "dark secrets thriller mystery psychological"),
    ("Dil Ne Kaha", "romantic heartbreak rediscovery emotional"),
    ("Mumbai Nights", "urban thriller mystery party secrets"),
    ("Hero Ban Gaya Villain", "action drama betrayal revenge stuntman"),
    ("Sapno Ki Baarish", "hope romance rain poetry emotions"),
    ("Mission Pariksha", "student drama UPSC preparation ambition stress"),
    ("Drama Queen Returns", "comedy comeback acting fame deception"),
    ("Andheron Mein Umeed", "drama emotional struggle hope poverty"),
    ("Kaun Hai Woh?", "horror suspense haunted secrets night"),
    ("Dilwale Detectives", "mystery romance comedy investigation"),
    ("Bhaag Rekha Bhaag", "sports drama race self-discovery journey"),
    ("Ek Tha Rapper", "music journey rejection struggle fame"),
    ("Magic Wali Mehfil", "fantasy drama musical magic festival"),
    ("Kya Love Algorithm Hai", "romantic comedy tech dating formula"),
    ("Zinda Dil", "inspirational youth cancer love hope"),
    ("Mohabbat Se Maut Tak", "tragic love revenge crime heartbreak"),
    ("Bollywood Express", "action entertainer drama train chase"),
    ("Ishq Market", "romance business manipulation betrayal"),
    ("Pyaar Ke Side Effects 2", "romcom marriage reality expectation"),
    ("Andheri Raat Ka Sapna", "thriller illusion nightmare dark secrets"),
    ("Metro Ka Musafir", "life journey commuter urban slice-of-life"),
    ("Raat Ki Rani", "mystery romance dream reality twist"),
    ("Dil Dhadakne Do 2", "family cruise drama comedy sequel"),
    ("The NRI Next Door", "romcom cross-culture humor family"),
    ("Bhookh", "social drama hunger poverty activism"),
    ("Bachpan Ki Duniya", "nostalgia school friends innocence drama"),
    ("Shaadi Se Pehle", "romantic comedy pre-wedding cold feet"),
    ("Dil Ka Rishta", "drama orphanage adoption love humanity"),
    ("Mission Rehmat", "spy action rescue mission family"),
    ("Gully Raftaar", "rap street performance dreams youth"),
    ("Chai Pe Charcha", "light comedy social satire politics"),
    ("Magic Mein Mohabbat", "fantasy love mystery stars dreams"),
    ("Bhaag Mohan Bhaag", "comedy self-discovery marathon twist"),
    ("Ishq Ki Fight", "action romance class conflict youth"),
    ("Pyaar Wali Programming", "romcom tech startup AI bots"),
    ("Bollywood Ka Boss", "industry satire fame greed rivalry"),
    ("Yeh Cinema Kya Hai?", "mockumentary art obsession film drama"),
    ("Zindagi Meter Down", "life taxi stories struggle humor"),
    ("Silent Superstar", "music drama mute singer performance"),
    ("Shaadi Shuffle", "romcom swapped weddings confusion chaos"),
    ("Dil Toh Code Karega", "romance coding introvert hackathon love"),
    ("Andheron Ka Sheher", "noir crime politics murder drama"),
    ("Ishq Express", "train romance destiny cute-strangers"),
    ("Bhaag Karan Bhaag", "sports dream underdog inspiring"),
    ("Zehar Ka Ishq", "toxic love betrayal manipulation"),
    ("Mumbai Ke Mautwale", "gang war action thriller revenge"),
    ("College Ki Kahani", "drama romance rivalry youth story"),
    ("Code Mera Dil", "tech drama heart AI mystery"),
    ("Monsoon Masala", "romcom rain chef restaurant food"),
    ("Kaun Banega Hero", "action spoof comedy reluctant hero"),
    ("Dil Ka Freelancer", "romantic story gig economy love"),
    ("Swag Se Shaadi", "romcom dance competition wedding prep"),
    ("Tum Hi Hack Ho", "cyber thriller romance ethics AI"),
    ("Raat Ka Reporter", "news thriller mystery corruption"),
    ("Zindagi Tandoor", "dark kitchen crime revenge love"),
    ("Saaz Aur Sazish", "musical thriller jealousy betrayal"),
    ("Pyaar In Pixels", "gaming romance digital age online love"),
    ("Shaadi Mein Twist", "comedy wedding suspense love triangle"),
    ("Metro Se Manzil", "commute journey life lessons friendship"),
    ("Khoobsurat Fraud", "romcom identity scam love story"),
    ("Jadoo Wali Jodi", "fantasy romance fairytale magic"),
    ("Aakhri Update", "tech thriller privacy hacking twist"),
    ("Dil Dosti Deadline", "youth friendship campus pressure coding"),
    ("Ghar Wapsi", "emotional family village return journey"),
    ("Main Hero Nahin", "anti-hero action drama emotional arc"),
    ("Silent Ishq", "mute love story emotions healing"),

]

titles=[title for title,desc in movies]
descriptions=[desc for title,desc in movies]

vectorizer=TfidfVectorizer()
tfidf_matrix=vectorizer.fit_transform(descriptions)

cosine_sim=cosine_similarity(tfidf_matrix)

def recommend(title,top_n=5):
    if title not in titles:
        return ["Movies not found in the database."]
    idx=titles.index(title)
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1], reverse=True)

    recommendations=[]
    for i,score in sim_scores[1:top_n+1]:
        recommendations.append(f"{titles[i]} (similarity:{score:.2f})")
    return recommendations

pygame.init()
WIDTH,HEIGHT=800,600
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Bollywood Movie Recommendation System")

WHITE=(255,255,255)
LIGHT_GRAY=(240,240,240)
GRAY=(200,200,200)
DARK_GRAY=(50,50,50)
BLACK=(0,0,0)
BLUE=(100,149,237)
DARK_BLUE=(70,130,180)
BG_TOP=(255,200,200)
BG_BOTTOM=(200,220,255)

font=pygame.font.SysFont("Arial",24)
big_font=pygame.font.SysFont("Arial",36,bold=True)
title_font=pygame.font.SysFont("Arial",28,bold=True)

input_box=pygame.Rect(50,80,700,45)
input_text=""
active=False

button_rect=pygame.Rect(330,140,140,45)

output_box=pygame.Rect(50,220,700,250)
recommendations=[]

def draw_rounded_rect(surface,color,rect,radius=10):
    """Draw a rounded rectangle."""
    pygame.draw.rect(surface,color,rect,border_radius=radius)

def draw_gradient_background(surface,top_color,bottom_color):
    """Draw a vertical gradient background."""
    for y in range(HEIGHT):
        blend=y/HEIGHT
        r=int(top_color[0]*(1-blend)+bottom_color[0]*blend)
        g=int(top_color[1]*(1-blend)+bottom_color[1]*blend)
        b=int(top_color[2]*(1-blend)+bottom_color[2]*blend)
        pygame.draw.line(surface,(r,g,b),(0,y),(WIDTH,y))

running=True
while running:
    draw_gradient_background(screen,BG_TOP,BG_BOTTOM)

    header=title_font.render("Bollywood Movie Recommendation System",True,DARK_GRAY)
    screen.blit(header,(WIDTH//2-header.get_width()//2,20))

    label=font.render("Enter Movie Title",True,BLACK)
    screen.blit(label,(50,50))

    draw_rounded_rect(screen,WHITE,input_box,radius=8)
    pygame.draw.rect(screen,BLUE if active else GRAY,input_box,2,border_radius=8)
    txt_surface=font.render(input_text,True,BLACK)
    screen.blit(txt_surface,(input_box.x+10,input_box.y+10))


    draw_rounded_rect(screen,DARK_BLUE,button_rect,radius=8)
    button_text=font.render("Recommend",True,WHITE)
    screen.blit(button_text,(button_rect.x+20,button_rect.y+10))

    draw_rounded_rect(screen,WHITE,output_box,radius=10)
    pygame.draw.rect(screen,DARK_BLUE,output_box,2,border_radius=10)

    y=output_box.y+20
    if recommendations:
        result_label=font.render("Top Recommendations:",True,BLACK)
        screen.blit(result_label,(output_box.x+20,y))
        y+=40
        for rec in recommendations:
            bullet_x=output_box.x+30
            bullet_y=y+10
            pygame.draw.circle(screen,DARK_BLUE,(bullet_x,bullet_y),5)
            rect_text=font.render(rec,True,DARK_GRAY)
            screen.blit(rect_text,(bullet_x+20,y))
            y+=35

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        elif event.type==pygame.MOUSEBUTTONDOWN:
            if input_box.collidepoint(event.pos):
                active=True
            else:
                active=False
            if button_rect.collidepoint(event.pos):
                if input_text.strip():
                    recommendations=recommend(input_text.strip())
                else:
                    recommendations=["Please enter a movie title."]
        elif event.type==pygame.KEYDOWN:
            if active:
                if event.key==pygame.K_RETURN:
                    if input_text.strip():
                        recommendations=recommend(input_text.strip())
                    else:
                        recommendations=["Please enter a movie title."]
                elif event.key==pygame.K_BACKSPACE:
                    input_text=input_text[:-1]
                else:
                    input_text+=event.unicode
    pygame.display.flip()

pygame.quit()
sys.exit()