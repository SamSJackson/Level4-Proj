from data.code.user_api.Generator import Generator


input_text = "The surface of Mars is barren and dry, with what little water there is tied up in icecaps or perhaps existing below the surface. But if you look closely at the surface, " \
            "you will see what looks like shorelines or canyons where massive "

model_name = "gpt2"
attempt_cuda = True
generator = Generator(model_name="gpt2", watermark_name="stanford",
                      attempt_cuda=True)

gamma = 0.4
delta = 5

content = generator.generate(input_text, gamma=gamma, delta=delta)

print(content)

