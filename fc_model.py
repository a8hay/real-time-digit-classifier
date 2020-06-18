import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
    """
    Builds a feedforward network with arbitary no of hidden layers.

    Arguments
    ----------
    input_size = integer, size of input layer
    output_size = integer, size of output layer
    hidden_layers = list of integers, size of different hidden layer
                    with the same order as list
    drop = integer between 0 and 1, drop probability for dropout
    """
    super().__init__()
    #input to hidden layer
    self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

    #add a variable no of more hidden layers
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

    self.output = nn.Linear(hidden_layers[-1], output_size)

    self.dropout = nn.Dropout(p=drop_p)
  
  def forward(self, x):
    """ Forward pass through the network return the logits """
    for each in self.hidden_layers:
      x = F.relu(each(x))
      x = self.dropout(x)
    
    x = self.output(x)

    return F.log_softmax(x, dim=1)

def validation(model, test_loader, criterion):
  accuracy = 0
  test_loss = 0

  for images, labels in test_loader:
    images = images.resize_(images.size()[0], 784)
    output = model.forward(images)
    test_loss += criterion(output, labels).item()

    # calculating the accuracy
    # model output is log softmax take exponetial to get probabilites
    ps = torch.exp(output)
    # compare the highest probabilites of each row with labels
    equals = ps.argmax(dim=1).eq(labels)
    # accuracy is correct prediction by total labels
    accuracy += equals.type_as(torch.FloatTensor()).mean()
  
  return test_loss, accuracy

def train(model, train_loader, test_loader, criterion, optimizer, epochs=5, print_every=40):
  steps = 0
  running_loss = 0

  for epoch in range(epochs):
    # model in training mode, dropout in on
    model.train()
    for images, labels in train_loader:
      step += 1

      #flatten the image into a vector
      images = images.resize_(images.size[0], 784)

      # clear the gradients
      optimizer.zero_grad()

      output = model.forward(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      #evaluate  every so often(according to value of print_every)
      if step % print_every == 0:
        # mode is inference, dropout is off
        model.eval()

        # turn off the gradients for inference will speed up the process
        with torch.no_grad():
          test_loss, accuracy = validation(model, test_loader, criterion)
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/print_every),
              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
              "Test Accuracy: {:.3f}.. ".format(accuracy/len(test_loader)))
        
        running_loss = 0
        # Make sure dropout and grads are on for training
        model.train()