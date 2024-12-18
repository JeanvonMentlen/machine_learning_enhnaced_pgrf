import torch
import numpy as np
from tqdm import tqdm

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def custom_accuracy(y_true, y_pred, tolerance=0.01):
    correct_predictions = torch.abs(y_true - y_pred) <= tolerance
    accuracy = torch.mean(correct_predictions.float())
    return accuracy

def train(model, train_dataloader, validation_dataloader, num_epochs, optimizer, loss_function, scheduler, logdir, model_name):
    prev_val_loss = 0.0
    prev_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_acc = float('inf')
    best_model_state = None
    training_history = list()
    validation_history = list()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for batch_idx, (batch_data, batch_labels) in tqdm(enumerate(train_dataloader)):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_data)

            # Calculate the loss
            loss = loss_function(predictions, batch_labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            training_accuracy = custom_accuracy(batch_labels, predictions, tolerance=0.005)

            running_loss += loss.item()
            running_accuracy += training_accuracy

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = running_accuracy / len(train_dataloader)

        training_history.append([epoch_loss, epoch_accuracy])
        # Print custom accuracy
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Custom Accuracy: {epoch_accuracy:.4f}")
        # Print MAE
        print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, MSE: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(validation_dataloader):
                val_outputs = model(val_inputs)
                val_loss = loss_function(val_outputs, val_targets)
                val_acc = custom_accuracy(val_outputs, val_targets, tolerance=0.001)

                val_running_acc += val_acc
                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(validation_dataloader)
        val_epoch_acc = val_running_acc / len(validation_dataloader)

        validation_history.append([val_epoch_loss, val_epoch_acc])

        print(f'--------------------------------------------------------------------')
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, delta: {val_epoch_loss - prev_val_loss:.4f}')
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Validation Accuracy: {val_epoch_acc:.4f}, delta: {val_epoch_acc - prev_val_acc:.4f}\n')
        prev_val_loss = val_epoch_loss
        prev_val_acc = val_epoch_acc

        scheduler.step()

        # Check if the current validation loss is better than the best validation loss
        if val_epoch_loss < best_val_loss or val_epoch_acc > best_val_acc:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            best_model_state = model.state_dict()
            print(f'Saving best model with Validation Loss: {best_val_loss}')

    training_history = np.array(training_history)
    validation_history = np.array(validation_history)

    model.load_state_dict(best_model_state)
    torch.save(model, logdir + '/' + model_name)

    return training_history, validation_history

def denoising_train(model, pretrained_model, train_dataloader, validation_dataloader, num_epochs, optimizer, loss_function, scheduler, logdir, model_name):
    prev_val_loss = 0.0
    prev_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_acc = float('inf')
    best_model_state = None
    training_history = list()
    validation_history = list()
    pretrained_model.eval()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for batch_idx, (batch_data, batch_labels) in tqdm(enumerate(train_dataloader)):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            prediction = pretrained_model(batch_data)

            # denoise
            denoised_prediction = model(batch_data, prediction)


            # Calculate the loss
            loss = loss_function(denoised_prediction, batch_labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            training_accuracy = custom_accuracy(batch_labels, denoised_prediction, tolerance=0.001)

            running_loss += loss.item()
            running_accuracy += training_accuracy

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = running_accuracy / len(train_dataloader)

        training_history.append([epoch_loss, epoch_accuracy])
        # Print custom accuracy
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Custom Accuracy: {epoch_accuracy:.4f}")
        # Print MAE
        print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, MSE: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(validation_dataloader):
                val_outputs = pretrained_model(val_inputs)
                denoised_val_outputs = model(val_inputs, val_outputs)
                val_loss = loss_function(denoised_val_outputs, val_targets)
                val_acc = custom_accuracy(denoised_val_outputs, val_targets, tolerance=0.001)

                val_running_acc += val_acc
                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(validation_dataloader)
        val_epoch_acc = val_running_acc / len(validation_dataloader)

        validation_history.append([val_epoch_loss, val_epoch_acc])

        print(f'--------------------------------------------------------------------')
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, delta: {val_epoch_loss - prev_val_loss:.4f}')
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Validation Accuracy: {val_epoch_acc:.4f}, delta: {val_epoch_acc - prev_val_acc:.4f}\n')
        prev_val_loss = val_epoch_loss
        prev_val_acc = val_epoch_acc

        scheduler.step()

        # Check if the current validation loss is better than the best validation loss
        if val_epoch_loss < best_val_loss or val_epoch_acc > best_val_acc:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            best_model_state = model.state_dict()
            print(f'Saving best model with Validation Loss: {best_val_loss}')

    training_history = np.array(training_history)
    validation_history = np.array(validation_history)

    model.load_state_dict(best_model_state)
    torch.save(model, logdir + '/' + model_name)

    return training_history, validation_history


def train_cgan(generator, discriminator, train_dataloader, validation_dataloader, num_epochs, optimizer_g, optimizer_d, loss_function, scheduler_g, scheduler_d, logdir, model_name):
    prev_val_loss = 0.0
    prev_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_acc = float('inf')
    best_generator_state = None
    best_discriminator_state = None
    training_history = list()
    validation_history = list()

    # Training Loop
    for epoch in range(num_epochs):
        running_g_loss = 0.0
        running_d_loss = 0.0
        for batch_idx, (curves, labels) in tqdm(enumerate(train_dataloader)):
            # Real and fake labels
            real_labels = torch.ones(labels.shape[0], labels.shape[1])
            fake_labels = torch.zeros(labels.shape[0], labels.shape[1])

            # Train Discriminator on real data
            outputs = discriminator(curves, labels)
            try:# Real curves and real labels
                d_loss_real = loss_function(outputs, real_labels)
            except:
                pass

            # Generate fake labels using the generator
            fake_labels_generated = generator(curves)  # Generate labels based on real curves

            # Train Discriminator on fake data
            outputs = discriminator(curves, fake_labels_generated.detach())  # Real curves and fake labels
            d_loss_fake = loss_function(outputs, fake_labels)

            # Combine losses and update Discriminator
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            outputs = discriminator(curves, fake_labels_generated)  # Real curves and fake labels
            g_loss = loss_function(outputs,
                                   real_labels)  # The generator wants the discriminator to think the labels are real

            # Update Generator
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()

        # Logging, saving, etc.
        epoch_g_loss = running_g_loss / len(train_dataloader)
        epoch_d_loss = running_d_loss / len(train_dataloader)

        training_history.append([epoch_g_loss, epoch_d_loss])
        # Print custom accuracy
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Generator Loss: {epoch_g_loss:.6f}")
        # Print MAE
        print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Discriminator Loss: {epoch_d_loss:.6f}")

        # Validation step
        generator.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(validation_dataloader):
                val_outputs = generator(val_inputs)
                val_loss = loss_function(val_outputs, val_targets)
                val_acc = custom_accuracy(val_outputs, val_targets, tolerance=0.001)

                val_running_acc += val_acc
                val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(validation_dataloader)
        val_epoch_acc = val_running_acc / len(validation_dataloader)

        validation_history.append([val_epoch_loss, val_epoch_acc])

        print(f'--------------------------------------------------------------------')
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, delta: {val_epoch_loss - prev_val_loss:.4f}')
        print(
            f'Epoch: {epoch + 1}/{num_epochs}, Validation Accuracy: {val_epoch_acc:.4f}, delta: {val_epoch_acc - prev_val_acc:.4f}\n')
        prev_val_loss = val_epoch_loss
        prev_val_acc = val_epoch_acc

        scheduler_g.step()
        scheduler_d.step()
        # Check if the current validation loss is better than the best validation loss
        if val_epoch_loss < best_val_loss or val_epoch_acc > best_val_acc:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            best_generator_state = generator.state_dict()
            best_discriminator_state = discriminator.state_dict()

            print(f'Saving best model with Validation Loss: {best_val_loss}')

    training_history = np.array(training_history)
    validation_history = np.array(validation_history)

    generator.load_state_dict(best_generator_state)
    discriminator.load_state_dict(best_discriminator_state)
    torch.save(best_generator_state, logdir + '/' + model_name)

    return training_history, validation_history

def test_model(forward_model, test_dataloader):
    forward_model.eval()

    val_output = []
    val_target = []
    with torch.no_grad():
        for val_batch_idx, (val_inputs, val_targets) in enumerate(test_dataloader):
            val_outputs = forward_model(val_inputs)

            val_output.append(val_outputs)
            val_target.append(val_targets)

    val_output = torch.cat(val_output, dim=0)
    val_target = torch.cat(val_target, dim=0)

    return val_output, val_target

def test_denoising_model(denoising_model, forward_model, test_dataloader):
    forward_model.eval()
    denoising_model.eval()

    val_output = []
    val_target = []
    with torch.no_grad():
        for val_batch_idx, (val_inputs, val_targets) in enumerate(test_dataloader):
            val_outputs = forward_model(val_inputs)
            denoised_val_outputs = denoising_model(val_inputs, val_outputs)

            val_output.append(denoised_val_outputs)
            val_target.append(val_targets)

    val_output = torch.cat(val_output, dim=0)
    val_target = torch.cat(val_target, dim=0)

    return val_output, val_target
