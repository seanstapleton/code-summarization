from torch import nn, optim

class Encoder(nn.Module):
    def __init__(self, input_size=100, embedding_size=50, hidden_size=50):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        # Can also try lstm
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size)

    def forward(self, input):
        embedded_out = self.embedding(input)
        output, hidden = self.gru(embedded_out,  hidden)

        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size=50, input_size=50, max_length=300 ):
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

def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, hidden_size):
    # zero out gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = torch.zeros(input.size(1), hidden_size)

    # For each timestep
    for t in range(input.size(0)):
        encoder_out, encoder_hidden = encoder(input[t], torch.zeros(hidden_sizes))

        # Need to index this
        encoder_outputs[t] = encoder_out()

def main():
    encoder = Encoder()
    decoder = Decoder()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

    loss_function = None

    train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function)


if __name__ == "__main__":
    main()
