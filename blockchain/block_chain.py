import hashlib
import json
import datetime
import os

class Block:
    """Individual block in the blockchain"""
    
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        """Calculate SHA-256 hash of block contents"""
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty):
        """Proof of work - mine block with given difficulty"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"Block mined: {self.hash}")

class Blockchain:
    """Blockchain implementation for storing detection results"""
    
    def __init__(self, difficulty=2, storage_file="results/blockchain_records.json"):
        self.chain = []
        self.difficulty = difficulty
        self.storage_file = storage_file
        self.create_genesis_block()
        self.load_from_file()
    
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(0, datetime.datetime.now().isoformat(), 
                            {"message": "Genesis Block - Intrusion Detection System"}, 
                            "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self):
        """Get the most recent block in the chain"""
        return self.chain[-1]
    
    def add_block(self, data):
        """Add a new block to the blockchain"""
        latest_block = self.get_latest_block()
        new_block = Block(latest_block.index + 1, 
                         datetime.datetime.now().isoformat(), 
                         data, 
                         latest_block.hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.save_to_file()
        return new_block
    
    def is_chain_valid(self):
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if hash is correct
            if current_block.hash != current_block.calculate_hash():
                print(f"Block {i} hash is invalid")
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.hash:
                print(f"Block {i} previous hash doesn't match")
                return False
            
            # Check if block meets difficulty requirement
            target = "0" * self.difficulty
            if current_block.hash[:self.difficulty] != target:
                print(f"Block {i} doesn't meet difficulty requirement")
                return False
        
        return True
    
    def save_to_file(self):
        """Save blockchain to file"""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'data': block.data,
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'nonce': block.nonce
            })
        
        with open(self.storage_file, 'w') as f:
            json.dump(chain_data, f, indent=2)
    
    def load_from_file(self):
        """Load blockchain from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    chain_data = json.load(f)
                
                # Rebuild chain
                loaded_chain = []
                for block_data in chain_data:
                    block = Block(
                        block_data['index'],
                        block_data['timestamp'],
                        block_data['data'],
                        block_data['previous_hash'],
                        block_data['nonce']
                    )
                    block.hash = block_data['hash']
                    loaded_chain.append(block)
                
                if len(loaded_chain) > len(self.chain):
                    self.chain = loaded_chain
            except Exception as e:
                print(f"Error loading blockchain: {e}")
    
    def get_detection_history(self):
        """Get all detection results from blockchain"""
        detections = []
        for block in self.chain[1:]:  # Skip genesis block
            if 'detection' in block.data:
                detections.append(block.data)
        return detections
    
    def get_statistics(self):
        """Get blockchain statistics"""
        total_blocks = len(self.chain)
        total_detections = len([b for b in self.chain[1:] if 'detection' in b.data])
        
        # Count attack types
        attack_counts = {}
        for block in self.chain[1:]:
            if 'detection' in block.data:
                for model_name, result in block.data['detection']['model_results'].items():
                    if 'threat_level' in result:
                        threat = result['threat_level']
                        attack_counts[threat] = attack_counts.get(threat, 0) + 1
        
        return {
            'total_blocks': total_blocks,
            'total_detections': total_detections,
            'attack_counts': attack_counts,
            'is_valid': self.is_chain_valid()
        }