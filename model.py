import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        # Can also try lstm
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size)

    def forward(self, input):
        embedded_out = self.embedding(input)
        output, hidden = self.gru(embedded_out,  hidden)

        return output, hidden

def Decoder(nn.Module):
    def __init__(self, hidden_size, input_size, max_length ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded_out = self.embedding(input)

        attn_weights = nn.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = nn.relu(output)
        output, hidden = self.gru(output, hidden)

        output = nn.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


def main():
    pass
    
if __name__ == "__main__":
    main()
