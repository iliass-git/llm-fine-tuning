def test_model(model, tokenizer, test_prompts):
    model.eval()
    for i, prompt in enumerate(test_prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
