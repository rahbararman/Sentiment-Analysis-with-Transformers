import torch
import data_loader_yelp
from model import SentimentAnalysisModel
import os

#Hyperparameters
learning_rate = 1e-3
n_embed = 8 
head_size = 4
n_layer = 4
batch_size = 8 
total_epochs = 4
num_classes = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
@torch.no_grad()
def calc_loss_sentiment(loader, m):
  m.eval()
  total_loss=0

  for x, y in loader:
    x, y = x.to(device), y.to(device)
    out, loss = model(x, y)
    total_loss += loss.item()

  m.train()
  return total_loss/len(loader)

def accuracy(m, loader):
  m.eval()  
  total_correct = 0
  total_instances = 0

  with torch.no_grad():
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      classifications = torch.argmax(m(x)[0], dim=1)

      correct_predictions = sum(classifications==y).item()
      total_correct+=correct_predictions
      total_instances+=len(x)
  return round(total_correct/total_instances, 3)

train_loader, train_data = data_loader_yelp.get_loader(os.path.join(os.getcwd(),"yelp_review_polarity_csv/train.csv"), num_workers=2)
test_loader, test_data = data_loader_yelp.get_loader(os.path.join(os.getcwd(),"yelp_review_polarity_csv/test.csv"),num_workers=2,vocab=train_data.get_vocab())

#create the model
model = SentimentAnalysisModel(len(train_data.vocab),n_embed, head_size, n_layer, num_classes).to(device)
#define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

epoch_test_losses = []
epoch_train_losses = []
for epoch in range(total_epochs):
  print('epoch number:' + str(epoch))
  #calculate loss at the beginning of the epoch
  test_loss = calc_loss_sentiment(test_loader, model)
  print('Test loss:' + str(test_loss))
  print('Test accuracy:' + str(accuracy(model, test_loader)))
  epoch_test_losses.append(test_loss)
  for i, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    out, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Final Test accuracy:' + str(accuracy(model, test_loader)))
