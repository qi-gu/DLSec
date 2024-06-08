model = 'ted_pretrained_v3'

# recipes: fgsm, pgd, genetic, cw, icw
# models:
# librispeech_pretrained_v3，
# an4_pretrained_v3，
# ted_pretrained_v3，
 
config = {
    'model': model,
    'goal': 'Hello world',
    'recipes': ['fgsm'],
    'device': 'cuda'
}