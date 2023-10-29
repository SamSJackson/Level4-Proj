from data.code.user_api.Generator import Generator
from data.code.user_api.Evaluator import Evaluator

input_text = "The surface of Mars is barren and dry, with what little water there is tied up in icecaps or perhaps existing below the surface. But if you look closely at the surface, " \
            "you will see what looks like shorelines or canyons where massive "

watermark_name = "stanford"
model_name = "gpt2"
tokenizer_name = "gpt2"
attempt_cuda = True
generator = Generator(model_name=model_name, tokenizer_name=tokenizer_name,
                      watermark_name=watermark_name,attempt_cuda=attempt_cuda)

evaluator = Evaluator(tokenizer_name=model_name, watermark_name=watermark_name,
                      attempt_cuda=attempt_cuda)

gamma = 0.4
delta = 20

content = generator.generate(input_text, gamma=gamma, delta=delta)
print(f"{input_text}-GENERATED CONTENT-{content}")

detection = evaluator.detect(content, gamma=gamma, delta=delta)
print(detection)