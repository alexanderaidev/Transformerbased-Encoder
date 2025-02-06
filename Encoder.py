import torch
import torch.nn as nn







class Encoder (nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Precompute die Positional Encoding Matrix
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)                                    # Gerade Indizes
        pe[:, 1::2] = torch.cos(position * div_term)                                    # Ungerade Indizes
        self.register_buffer('pe', pe.unsqueeze(0))                                     # Shape: (1, max_seq_len, embedding_dim)
        
        
        
        # Multihead1
        self.multihead_attention_layer1 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward1 
        self.ffn1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead2
        self.multihead_attention_layer2 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward2
        self.ffn2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead3
        self.multihead_attention_layer3 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward3
        self.ffn3 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead4
        self.multihead_attention_layer4 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward4
        self.ffn4 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead5
        self.multihead_attention_layer5 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward5
        self.ffn5 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead6
        self.multihead_attention_layer6 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward6
        self.ffn6 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multiwear7
        self.multihead_attention_layer7 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward7
        self.ffn7 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead8
        self.multihead_attention_layer8 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward8
        self.ffn8 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead9
        self.multihead_attention_layer9 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward9
        self.ffn9 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead10
        self.multihead_attention_layer10 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward10
        self.ffn10 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead11
        self.multihead_attention_layer11 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward11
        self.ffn11 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead12
        self.multihead_attention_layer12 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward12
        self.ffn12 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead13
        self.multihead_attention_layer13 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward13
        self.ffn13 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead14
        self.multihead_attention_layer14 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward14
        self.ffn14 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead15
        self.multihead_attention_layer15 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward15
        self.ffn15 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead16
        self.multihead_attention_layer16 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward16
        self.ffn16 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead17
        self.multihead_attention_layer17 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward17
        self.ffn17 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead18
        self.multihead_attention_layer18 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward18
        self.ffn18 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead19
        self.multihead_attention_layer19 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward19
        self.ffn19 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead20
        self.multihead_attention_layer20 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward20
        self.ffn20 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        # Multihead21
        self.multihead_attention_layer21 = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, batch_first = True)
        
        # Feedforward21
        self.ffn21 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),  # Erhöhung der Dimensionen
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)  # Rückkehr zur ursprünglichen Dimension
        )
        
        
        # Lineare schicht, um für das Training den output in die vocabgröße zu bringen.
        self.output_linear = nn.Linear(embedding_dim, vocab_size)
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.layer_norm4 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm5 = nn.LayerNorm(embedding_dim)
        self.layer_norm6 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm7 = nn.LayerNorm(embedding_dim)
        self.layer_norm8 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm9 = nn.LayerNorm(embedding_dim)
        self.layer_norm10 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm11 = nn.LayerNorm(embedding_dim)
        self.layer_norm12 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm13 = nn.LayerNorm(embedding_dim)
        self.layer_norm14 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm15 = nn.LayerNorm(embedding_dim)
        self.layer_norm16 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm17 = nn.LayerNorm(embedding_dim)
        self.layer_norm18 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm19 = nn.LayerNorm(embedding_dim)
        self.layer_norm20 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm21 = nn.LayerNorm(embedding_dim)
        self.layer_norm22 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm23 = nn.LayerNorm(embedding_dim)
        self.layer_norm24 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm25 = nn.LayerNorm(embedding_dim)
        self.layer_norm26 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm27 = nn.LayerNorm(embedding_dim)
        self.layer_norm28 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm29 = nn.LayerNorm(embedding_dim)
        self.layer_norm30 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm31 = nn.LayerNorm(embedding_dim)
        self.layer_norm32 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm33 = nn.LayerNorm(embedding_dim)
        self.layer_norm34 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm35 = nn.LayerNorm(embedding_dim)
        self.layer_norm36 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm37 = nn.LayerNorm(embedding_dim)
        self.layer_norm38 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm39 = nn.LayerNorm(embedding_dim)
        self.layer_norm40 = nn.LayerNorm(embedding_dim)
        
        self.layer_norm41 = nn.LayerNorm(embedding_dim)
        self.layer_norm42 = nn.LayerNorm(embedding_dim)

     
      
    def forward(self, x):
        
        key_padding_mask = (x == 0).to(torch.float32)
        key_padding_mask[key_padding_mask == 1] = float("-inf")
        padding_mask = (key_padding_mask == 0).unsqueeze(-1)
        
        #print("Maske im Encoder: ", key_padding_mask)
        #Verhindern von vollständig maskierter Sequenz, um NaN zu vermeiden
        
        #if key_padding_mask.dim() == 1:
           # if key_padding_mask.all():
                #key_padding_mask[:, 0] = False
                
        #elif key_padding_mask.dim() == 2:
            #if key_padding_mask.all(dim=1).any():
               # key_padding_mask[:, 0] = False
        
        #print(x)    
        #print("Encoder Padding Maske: ", key_padding_mask)
                
                
        x = self.embedding(x) + self.pe[:, : x.size(1), :]
        #x = x.permute(1, 0,2)
       

        # Block 1
        attn_output1, attn_weights1 = self.multihead_attention_layer1(x, x, x, key_padding_mask = key_padding_mask)
        
        attn_output1 = attn_output1 * padding_mask
        
        x = self.layer_norm1(attn_output1 + x)                                          # Residual Connection
        x = x * padding_mask
        
        ffn_output1 = self.ffn1(x)
        ffn_output1 = ffn_output1 * padding_mask
        
        x = self.layer_norm2(ffn_output1 + x)
        x = x * padding_mask

        # Block 2
        attn_output2, attn_weights2 = self.multihead_attention_layer2(x, x, x, key_padding_mask = key_padding_mask)
        attn_output2 = attn_output2 * padding_mask
        
        x = self.layer_norm3(attn_output2 + x)
        x = x * padding_mask
        
        ffn_output2 = self.ffn2(x)
        ffn_output2 = ffn_output2 * padding_mask
        
        x = self.layer_norm4(ffn_output2 + x)
        x = x * padding_mask
        
        # Block 3
        attn_output3, attn_weights3 = self.multihead_attention_layer3(x, x, x, key_padding_mask = key_padding_mask)
        attn_output3 = attn_output3 * padding_mask
        
        x = self.layer_norm5(attn_output3 + x)
        x = x * padding_mask
        
        ffn_output3 = self.ffn3(x)
        ffn_output3 = ffn_output3 * padding_mask
        
        x = self.layer_norm6(ffn_output3 + x)
        x = x * padding_mask
        
        # Block 4
        attn_output4, attn_weights4 = self.multihead_attention_layer4(x, x, x, key_padding_mask = key_padding_mask)
        attn_output4 = attn_output4 * padding_mask
        
        x = self.layer_norm7(attn_output4 + x)
        x = x * padding_mask
        
        ffn_output4 = self.ffn4(x)
        ffn_output4 = ffn_output4 * padding_mask
        x = self.layer_norm8(ffn_output4 + x)
        x = x * padding_mask
        
        # Block 5
        attn_output5, attn_weights5 = self.multihead_attention_layer5(x, x, x,key_padding_mask = key_padding_mask)
        attn_output5 = attn_output5 * padding_mask
        
        x = self.layer_norm9(attn_output5 + x)
        x = x * padding_mask
        
        ffn_output5 = self.ffn5(x)
        ffn_output5 = ffn_output5 * padding_mask
        
        x = self.layer_norm10(ffn_output5 + x)
        x = x * padding_mask
        
        # Block 6
        attn_output6, attn_weights6 = self.multihead_attention_layer6(x, x, x, key_padding_mask = key_padding_mask)
        attn_output6 = attn_output6 * padding_mask
        
        x = self.layer_norm11(attn_output6 + x)
        x = x * padding_mask
        
        ffn_output6 = self.ffn6(x)
        ffn_output6 = ffn_output6 * padding_mask
        
        x = self.layer_norm12(ffn_output6 + x)
        x = x * padding_mask
        
        # Block 7
        attn_output7, attn_weights7 = self.multihead_attention_layer7(x, x, x, key_padding_mask = key_padding_mask)
        attn_output7 = attn_output7 * padding_mask
        
        x = self.layer_norm13(attn_output7 + x)
        x = x * padding_mask
        
        ffn_output7 = self.ffn7(x)
        ffn_output7 = ffn_output7 * padding_mask
        
        x = self.layer_norm14(ffn_output7 + x)
        x = x * padding_mask
        
        # Block 8
        attn_output8, attn_weights8 = self.multihead_attention_layer8(x, x, x, key_padding_mask = key_padding_mask)
        attn_output8 = attn_output8 * padding_mask
        
        x = self.layer_norm15(attn_output8 + x)
        x = x * padding_mask
        
        ffn_output8 = self.ffn8(x)
        ffn_output8 = ffn_output8 * padding_mask
        
        x = self.layer_norm16(ffn_output8 + x)
        x = x * padding_mask
        
        # Block 9
        attn_output9, attn_weights9 = self.multihead_attention_layer9(x, x, x, key_padding_mask = key_padding_mask)
        attn_output9 = attn_output9 * padding_mask
        
        x = self.layer_norm17(attn_output9 + x)
        x = x * padding_mask
        
        ffn_output9 = self.ffn9(x)
        ffn_output9 = ffn_output9 * padding_mask
        
        x = self.layer_norm18(ffn_output9 + x)
        x = x * padding_mask
        
        # Block 10
        attn_output10, attn_weights10 = self.multihead_attention_layer10(x, x, x, key_padding_mask = key_padding_mask)
        attn_output10 = attn_output10 * padding_mask
        
        x = self.layer_norm19(attn_output10 + x)
        x = x * padding_mask
        
        ffn_output10 = self.ffn10(x)
        ffn_output10 = ffn_output10 * padding_mask
        x = self.layer_norm20(ffn_output10 + x)
        x = x * padding_mask
        
        # Block 11
        attn_output11, attn_weights11 = self.multihead_attention_layer11(x, x, x, key_padding_mask = key_padding_mask)
        attn_output11 = attn_output11 * padding_mask
        
        x = self.layer_norm21(attn_output11 + x)
        x = x * padding_mask
        
        ffn_output11 = self.ffn11(x)
        ffn_output11 = ffn_output11 * padding_mask
        
        x = self.layer_norm22(ffn_output11 + x)
        x = x * padding_mask
        
        # Block 12
        attn_output12, attn_weights12 = self.multihead_attention_layer12(x, x, x, key_padding_mask = key_padding_mask)
        attn_output12 = attn_output12 * padding_mask
        
        x = self.layer_norm23(attn_output12 + x)
        x = x * padding_mask
        
        ffn_output12 = self.ffn12(x)
        ffn_output12 = ffn_output12 * padding_mask
        
        x = self.layer_norm24(ffn_output12 + x)
        x = x * padding_mask
        
        # Block 13
        attn_output13, attn_weights13 = self.multihead_attention_layer13(x, x, x, key_padding_mask = key_padding_mask)
        attn_output13 = attn_output13 * padding_mask
        
        x = self.layer_norm25(attn_output13 + x)
        x = x * padding_mask
        
        ffn_output13 = self.ffn13(x)
        ffn_output13 = ffn_output13 * padding_mask
        
        x = self.layer_norm26(ffn_output13 + x)
        x = x * padding_mask
        
        # Block 14
        attn_output14, attn_weights14 = self.multihead_attention_layer14(x, x, x, key_padding_mask = key_padding_mask)
        attn_output14 = attn_output14 * padding_mask
        
        x = self.layer_norm27(attn_output14 + x)
        x = x * padding_mask
        
        ffn_output14 = self.ffn14(x)
        ffn_output14 = ffn_output14 * padding_mask
        
        x = self.layer_norm28(ffn_output14 + x)
        x = x * padding_mask
        
        # Block 15
        attn_output15, attn_weights15 = self.multihead_attention_layer15(x, x, x, key_padding_mask = key_padding_mask)
        attn_output15 = attn_output15 * padding_mask
        
        x = self.layer_norm29(attn_output15 + x)
        x = x * padding_mask
        
        ffn_output15 = self.ffn15(x)
        ffn_output15 = ffn_output15 * padding_mask
        
        x = self.layer_norm30(ffn_output15 + x)
        x = x * padding_mask
        
        # Block 16
        attn_output16, attn_weights16 = self.multihead_attention_layer16(x, x, x, key_padding_mask = key_padding_mask)
        attn_output16 = attn_output16 * padding_mask
        
        x = self.layer_norm31(attn_output16 + x)
        x = x * padding_mask
        
        ffn_output16 = self.ffn16(x)
        ffn_output16 = ffn_output16 * padding_mask
        
        x = self.layer_norm32(ffn_output16 + x)
        x = x * padding_mask
        
        # Block 17
        attn_output17, attn_weights17 = self.multihead_attention_layer17(x, x, x, key_padding_mask = key_padding_mask)
        attn_output17 = attn_output17 * padding_mask
        
        x = self.layer_norm33(attn_output17 + x)
        x = x * padding_mask
        
        ffn_output17 = self.ffn17(x)
        ffn_output17 = ffn_output17 * padding_mask
        
        x = self.layer_norm34(ffn_output17 + x)
        x = x * padding_mask
        
        # Block 18
        attn_output18, attn_weights18 = self.multihead_attention_layer18(x, x, x, key_padding_mask = key_padding_mask)
        attn_output18 = attn_output18 * padding_mask
        
        x = self.layer_norm35(attn_output18 + x)
        x = x * padding_mask
        
        ffn_output18 = self.ffn18(x)
        ffn_output18 = ffn_output18 * padding_mask
        
        x = self.layer_norm36(ffn_output18 + x)
        x = x * padding_mask
        
        # Block 19
        attn_output19, attn_weights19 = self.multihead_attention_layer19(x, x, x, key_padding_mask = key_padding_mask)
        attn_output19 = attn_output19 * padding_mask
        
        x = self.layer_norm37(attn_output19 + x)
        x = x * padding_mask
        
        ffn_output19 = self.ffn19(x)
        ffn_output19 = ffn_output19 * padding_mask
        
        x = self.layer_norm38(ffn_output19 + x)
        x = x * padding_mask
        
        # Block 20
        attn_output20, attn_weights20 = self.multihead_attention_layer20(x, x, x, key_padding_mask = key_padding_mask)
        attn_output20 = attn_output20 * padding_mask
        
        x = self.layer_norm39(attn_output20 + x)
        x = x * padding_mask
        
        ffn_output20 = self.ffn20(x)
        ffn_output20 = ffn_output20 * padding_mask
        
        x = self.layer_norm40(ffn_output20 + x)
        x = x * padding_mask
        
        # Block 21
        attn_output21, attn_weights21 = self.multihead_attention_layer21(x, x, x, key_padding_mask = key_padding_mask)
        attn_output21 = attn_output21 * padding_mask
        
        x = self.layer_norm41(attn_output21 + x)
        x = x * padding_mask
        
        ffn_output21 = self.ffn21(x)
        ffn_output21 = ffn_output21 * padding_mask
        
        x = self.layer_norm42(ffn_output21 + x)
        x = x * padding_mask
        
        # Eine lineare Schicht, um von Embedding-Dimension auf Vokabulargröße zu projizieren
        x = self.output_linear(x)  # Shape nach Umwandlung: [3, 512, 30000]

                
        return x, key_padding_mask
    

