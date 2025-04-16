import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
sys.path.append("/data1/wangyan/ModelFinetuning/data_manage")
from data_manage import DataManage
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from torch import LongTensor
import torch.optim as optim

class MLP(nn.Module):
    """
    addition MLP for docs categories classification
    a 3-layer MLP used to map the last hiddden layer of token [DOC]
    into probability distribution of doc labels (5 labels: Golden, Distracting, ...)
    """
    def __init__(self, input_dim: int, drop_out_prob: float|None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 30
        self.output_dim = 5

        # Define the 3-layer MLP using nn.Sequential (without softmax)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out_prob) if drop_out_prob else nn.Identity(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, input_state: torch.Tensor):
        """
        Args:
            input_state (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim=5).
        """
        # Ensure data type alignment
        if input_state.dtype != self.mlp[0].weight.dtype:
            input_state = input_state.to(self.mlp[0].weight.dtype)

        # Forward pass through the MLP
        output = self.mlp(input_state)
        return output


class CLMWithMLP(PreTrainedModel):
    """
    A wrapper class that combines a causal language model (CLM) with an additional MLP.
    1. The MLP takes the hidden state of the [DOC] token from the last layer of the CLM
    and outputs a probability distribution over document categories.
    2. The LLM's original output is also returned for multi-task learning.
    """
    def __init__(self, 
                 init: bool, # "initialization" or "reload_model"
                 device: str,
                 clm_model_path: str|None =None, 
                 drop_out_prob: float|None =None,
                 clm: AutoModelForCausalLM|None =None,
                 tokenizer: AutoTokenizer|None =None,
                 mlp: MLP|None =None):
        """
        Case 1: Regular Initialization: loading the LLM from pretrained model, initilizing the MLP
            Args:
                device (str): Device to load the model on ('cpu' or 'cuda:0' ... )
                clm_model_path (str): Pretrained CLM model path
                drop_out_prob (float|None): Dropout probability for the MLP. If None, no dropout is applied
                clm (None)
                tokenizer (None)
                mlp (None)

        Case 2: Loading models from external sources (for load_madel method)
            Args:
                device (str): Device to load the model on ('cpu' or 'cuda:0' ... )
                clm_model_path (str): reloaded CLM model path
                drop_out_prob (None)
                clm (AutoModelForCausalLM)
                tokenizer (AutoTokenizer)
                mlp (MLP)
        """
        self._device = device
        self.doc_token = "[DOC]"

        # Case 1: Regular Initialization
        if init==True:
            super().__init__(AutoModelForCausalLM.from_pretrained(clm_model_path).config)
            
            # Load the pretrained causal language model
            self.clm = AutoModelForCausalLM.from_pretrained(
                clm_model_path,
                torch_dtype="bfloat16",
                device_map=self._device,
                attn_implementation="eager")
            
            # Define the MLP for document classification
            self.mlp = MLP(input_dim=self.clm.config.hidden_size, drop_out_prob=drop_out_prob).to(self._device)
            
            # Tokenizer to map [DOC] token to its corresponding ID
            self.tokenizer = AutoTokenizer.from_pretrained(clm_model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # add special token into tokenizer
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.doc_token]})
            # reorganizer the size of embedding
            self.clm.resize_token_embeddings(len(self.tokenizer))

        # Case 2: ReLoading models from external sources
        if init==False:
            super().__init__(clm.config)
            self.clm = clm
            self.mlp = mlp
            self.tokenizer = tokenizer

        self.doc_token_id: int
        self.doc_token_id = self.tokenizer.convert_tokens_to_ids(self.doc_token)
    
    def forward(self, input_text: str, output_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CLM and MLP.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size=1, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size=1, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mlp_output: Output tensor of shape (batch_size=10, output_dim=5), representing the logits for each category.
                - llm_output: Original LLM output (e.g., logits or hidden states).
        """
        model_inputs = self.tokenizer([input_text+output_text], return_tensors="pt").to(self._device)

        # 将 input 部分的 token 标记为 -100，以便忽略它们的 loss
        labels = model_inputs["input_ids"].clone()
        input_text_length = len(self.tokenizer(input_text, return_tensors="pt")["input_ids"][0])
        labels[0, :input_text_length] = -100  # 忽略 input 部分的 loss

        # Pass input through the CLM to get the last hidden states and logits
        clm_outputs = self.clm(
            **model_inputs,
            labels=labels,
            output_hidden_states=True)
        last_hidden_state = clm_outputs.hidden_states[-1]  # Shape: (batch_size=1, seq_len, hidden_size)
        # print("last_hidden_state.size()", last_hidden_state.size())
        llm_logits = clm_outputs.logits  # Shape: (batch_size=1, seq_len, vocab_size)
        clm_loss = clm_outputs.loss
        
        # Find the position of the [DOC] token in each sequence
        # eg: input_ids = torch.tensor([[101, 202, 505, 505]])
        #     self.doc_token_id = 505
        #     doc_token_positions = (tensor([0, 0]), tensor([2, 3]))
        doc_token_positions = (model_inputs['input_ids'] == self.doc_token_id).nonzero(as_tuple=True)
        
        # Extract the hidden state corresponding to the [DOC] token
        # Shape: ([DOC]_num = 5, hidden_size)
        doc_hidden_states = last_hidden_state[doc_token_positions[0], doc_token_positions[1]]
        
        # Pass the [DOC] token's hidden state through the MLP
        mlp_output = self.mlp(doc_hidden_states)  # Shape: ([DOC]_num = 5, output_dim=5)
        
        # Return both outputs
        return mlp_output, llm_logits, clm_loss
    
    def generate(self, input_text: str):
        """
        Generate the output through the input token IDs and attention mask.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (1, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (1, seq_len).

        Returns:
            output_text (str): Generated text.
        """
        # transfer input_text into input_token_id_tensor & attention mask
        # eg: {'input_ids': tensor([[128000, 271, ...]]), 'attention_mask': tensor([[1, 1, ...]])}
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self._device)
        input_ids = model_inputs.input_ids
        input_ids_length = input_ids.shape[1]


        # generate output_token_id_tensor through input_token_id_tensor
        outputs_ids = self.clm.generate(
            **model_inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )[0, input_ids_length:]  # Exclude the input part
        outputs_ids: LongTensor

        # transfer output_token_id_tensor into output_text
        print("outputs_ids: ", outputs_ids)
        output_text = self.tokenizer.decode(
            outputs_ids,
            # skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return output_text
    
    def save_model(self, save_dir: str):
        """
        Save the entire model (both CLM with tokenizer and MLP) to the specified directory.

        Args:
            save_dir (str): Directory where the model will be saved.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the CLM model
        clm_save_path = os.path.join(save_dir, "clm_model")
        self.clm.save_pretrained(clm_save_path)

        # Save the tokenizer
        tokenizer_save_path = os.path.join(save_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_save_path)

        # Save the MLP
        mlp_save_path = os.path.join(save_dir, "mlp.pth")
        torch.save(self.mlp.state_dict(), mlp_save_path)

        print(f"Model saved successfully to {save_dir}")
        return
    
    # This is a class method: it can be use through class itself 
    # rather than the instance ofr the class
    # eg: CLMWithMLP.load_model(...)
    @classmethod
    def load_model(cls, 
                   load_dir: str, 
                   device: str, 
                   drop_out_prob: float|None):
        """
        Load the model (both CLM and MLP) from the specified directory.

        Args:
            load_dir (str): Directory from which the model will be loaded.
            device (str): Device to load the model on ('cpu' or 'cuda').
            clm_model_path (Optional[str]): Path to the pretrained CLM model if not loading from `load_dir`.
            drop_out_prob (Optional[float]): Dropout probability for the MLP. If None, no dropout is applied.

        Returns:
            CLMWithMLP: Loaded instance of the CLMWithMLP model.
        """
        # Load the CLM model
        clm_load_path = os.path.join(load_dir, "clm_model")
        clm = AutoModelForCausalLM.from_pretrained(
            clm_load_path,
            torch_dtype="bfloat16",
            device_map=device,
            attn_implementation="eager")
        print("\nCLM model loaded successfully.\n")

        # Load the tokenizer
        tokenizer_load_path = os.path.join(load_dir, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)

        # Load the MLP
        mlp_load_path = os.path.join(load_dir, "mlp.pth")
        mlp = MLP(input_dim=clm.config.hidden_size, drop_out_prob=drop_out_prob)
        mlp.load_state_dict(torch.load(mlp_load_path, map_location=device))
        mlp.to(device)
        print("\nMLP model loaded successfully.\n")

        # Initialize the CLMWithMLP instance with preloaded components
        model_CLMWithMLP = cls(init=False, 
                               clm_model_path=clm_load_path,
                               device=device, 
                               clm=clm, 
                               tokenizer=tokenizer, 
                               mlp=mlp)
        
        print(f"Model loaded successfully from {load_dir}")
        return model_CLMWithMLP
    

class Training:
    """
    A training class for the CLMWithMLP model.
    This class handles the training loop, loss computation, and optimization.
    """
    def __init__(self,
                 model_path: str,
                 trained_model_save_dir: str,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 learning_rate: float,
                 weight_decay: float,
                 loss_weight: dict, # example loss_weight = {"max": 1.0, "norm": 1.0, "cls": 1.0}
                 device: str,
                 epochs: int):
        """
        Initialize the Training class.

        Args:
            model (CLMWithMLP): The CLMWithMLP model to be trained.
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for regularization.
            device (str): Device to run the training ('cpu' or 'cuda').
            epochs (int): Number of training epochs.
        """
        # initialize the model
        self.model = CLMWithMLP(init=True,
                                device=device,
                                clm_model_path=model_path,
                                drop_out_prob=None)
        
        # initialize the dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # initialize the training parameters
        self.device = device
        self.epochs = epochs
        self.trained_model_save_dir = trained_model_save_dir

        # classification loss functions
        self.mlp_loss_fn = nn.CrossEntropyLoss()  # For classification task

        # weights for the loss function
        # example loss_weight = {"max": 1.0, "norm": 1.0, "cls": 1.0}
        self.w_max = loss_weight["max"]
        self.w_norm = loss_weight["norm"]
        self.w_cls = loss_weight["cls"]

        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.model.clm.parameters()) + list(self.model.mlp.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            float: Average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        # self.train_dataloader 每迭代一次进度条+1
        progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)

        for batch in progress_bar:
            batch_size = len(batch['input_text_list_batch'])
            for i in range(batch_size):
                # process a sample with several groups
                output_text = batch['output_text_batch'][i]
                output_text: str
                group_size = len(batch['input_text_list_batch'][i])
                combined_mlp_loss = 0
                combined_clm_loss = []
                for j in range(group_size):
                    # process a group in the sample
                    input_text = batch['input_text_list_batch'][i][j]
                    input_text: str
                    doc_labels = batch["output_class_num_list_batch"][i][j]
                    # -- category labels for the [DOC] token for 0-4
                    doc_labels: torch.Tensor # Shape: ([DOC_num]) 

                    # Forward pass
                    # mlp_output Shape: ([DOC_num, output_dim=doc_type_num=5])
                    mlp_output, llm_logits, clm_loss = self.model.forward(input_text, output_text)
                    print("clm_loss", clm_loss)
                    combined_clm_loss.append(clm_loss)

                    # Compute MLP loss (classification task)
                    # 分别求5个DOC的loss再求平均
                    mlp_loss = self.mlp_loss_fn(mlp_output, doc_labels.to(self.device))
                    print("mlp_loss", mlp_loss)
                    combined_mlp_loss += mlp_loss

                # 将列表中的张量堆叠成一个张量
                combined_clm_loss_tensor = torch.stack(combined_clm_loss)
                # 使用 PyTorch 的 max 和 min 函数
                max_clm_loss = torch.max(combined_clm_loss_tensor)
                min_clm_loss = torch.min(combined_clm_loss_tensor)
                # normalization
                norm_clm_loss = (max_clm_loss - min_clm_loss) ** 2
                # Combine losses
                combined_loss = self.w_max*max_clm_loss + self.w_norm*norm_clm_loss + self.w_cls*combined_mlp_loss
                # Update total loss
                total_loss += combined_loss.item()

                # gradient clear
                self.optimizer.zero_grad()
                # back propogation
                combined_loss.backward()
                # optimization (update the parameter)
                self.optimizer.step()

                # 显示当前sample的loss和average_loss_per_sample
                progress_bar.set_postfix(loss=combined_loss.item(),
                                         average_loss_per_sample=total_loss/len(self.train_dataloader*batch_size))

        # average_loss_per_batch over the epoch
        return total_loss/len(self.train_dataloader)

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model on the validation set.

        Returns:
            Tuple[float, float]: Average MLP loss and LLM loss on the validation set.
        """
        self.model.eval()
        total_mlp_loss = 0.0
        total_llm_loss = 0.0
        progress_bar = tqdm(self.val_dataloader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                batch_size = len(batch['input_text_list_batch'])
                for i in range(batch_size):
                    # process a sample with several groups
                    output_text = batch['output_text_batch'][i]
                    output_text: str
                    group_size = len(batch['input_text_list_batch'][i])
                    combined_mlp_loss = 0
                    combined_clm_loss = []
                    for j in range(group_size):
                        # process a group in the sample
                        input_text = batch['input_text_list_batch'][i][j]
                        input_text: str
                        doc_labels = batch["output_class_num_list_batch"][i][j]
                        # -- category labels for the [DOC] token for 0-4
                        doc_labels: torch.Tensor # Shape: ([DOC_num]) 

                        # Forward pass
                        # mlp_output Shape: ([DOC_num, output_dim=doc_type_num=5])
                        # clm_loss: float标量
                        mlp_output, llm_logits, clm_loss = self.model.forward(input_text, output_text)
                        # print("clm_loss", clm_loss)
                        combined_clm_loss.append(clm_loss.item())

                        # Compute MLP loss (classification task)
                        # 分别求5个DOC的loss再求平均
                        # mlp_loss: Tensor 标量
                        mlp_loss = self.mlp_loss_fn(mlp_output, doc_labels.to(self.device))
                        # print("mlp_loss", mlp_loss)
                        combined_mlp_loss += mlp_loss.item()

                    # 使用 PyTorch 的 max 和 min 函数
                    max_clm_loss = max(combined_clm_loss)
                    min_clm_loss = min(combined_clm_loss)
                    # normalization
                    norm_clm_loss = (max_clm_loss - min_clm_loss) ** 2
                    
                    total_llm_loss += (max_clm_loss + norm_clm_loss)
                    total_mlp_loss += combined_mlp_loss

        # average_loss_per_batch
        avg_mlp_loss = total_mlp_loss/len(self.val_dataloader)
        avg_llm_loss = total_llm_loss/len(self.val_dataloader)
        return avg_mlp_loss, avg_llm_loss

    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self.train_epoch()
            val_mlp_loss, val_llm_loss = self.validate()

            print(f"Train Loss (average_loss_per_sample): {train_loss:.4f}, "
                  f"Val MLP Loss: {val_mlp_loss:.4f}, "
                  f"Val LLM Loss: {val_llm_loss:.4f}")

            # Save the model at the end of each epoch
            save_dir = os.path.join(self.trained_model_save_dir, f"epoch_{epoch + 1}")
            self.model.save_model(save_dir)

            print(f"Model saved to {save_dir}\n")


# test
if __name__ == "__main__":

    MODEL_PATH_DICT = {"Qwen2.5-7B": "/datanfs2/zyx/model_cache/qwen/Qwen2___5-7B-Instruct", 
                       "Llama-3.1-8B": "/datanfs2/zyx/model_cache/LLM-Research/Meta-Llama-3___1-8B-Instruct"}


    dataset_path = "/data1/wangyan/ModelFinetuning/dataset/origin_dataset/train.json"
    dataset_cache_dir_path = "/data1/wangyan/ModelFinetuning/dataset/dataset_cache"
    

    clm_model_path = MODEL_PATH_DICT["Qwen2.5-7B"]
    trained_model_save_dir = "/datanfs2/zyx/model_cache/qwen/Qwen2___5-7B-Instruct_finetuning"
    
    train_data_manage = DataManage("Qwen2.5-7B", clm_model_path, dataset_path, "train", dataset_cache_dir_path)
    train_dataloader = train_data_manage.dataloader(batch_size=1, shuffle=False)

    val_data_manage = DataManage("Qwen2.5-7B", clm_model_path, dataset_path, "dev", dataset_cache_dir_path)
    val_dataloader = val_data_manage.dataloader(batch_size=1, shuffle=False)

    learning_rate = 1e-5
    weight_decay = 1e-5
    device = "cuda:0"
    epochs = 5
    loss_weight = {"max": 0.5, "norm": 0.2, "cls": 0.3}

    train = Training(clm_model_path, 
                    trained_model_save_dir,
                    train_dataloader,
                    val_dataloader,
                    learning_rate,
                    weight_decay,
                    loss_weight,
                    device,
                    epochs)
    
    a, b = train.train_epoch()
    print("a, b", a, b)