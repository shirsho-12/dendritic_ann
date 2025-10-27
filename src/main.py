from trainer import Trainer


def main():
    trainer = Trainer(
        learning_rate=0.001,
        dataset_name="mnist",
        data_root="../data",
        add_noise=True,
        sigma=0.1,
        batch_size=64,
        weight_decay=0.0001,
        seed=42,
        shuffle=True,
        perturb=False,
    )
    trainer.train(num_epochs=10)
    accuracy = trainer.evaluate()
    print(f"Model accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
