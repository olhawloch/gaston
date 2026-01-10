"""
Utilities for building and trimming conversation trees.
"""

from typing import Dict, List, Optional

def combine_nodes_to_tree(data_list: list[dict], max_depth=None) -> dict | None:
    """
    Converts a list of dictionaries with 'id' and 'parent_id' fields into a
    nested tree structure, assuming there is only one root node in the list
    that has 'node_type' of 'submission'.
    Limits the depth of the tree.

    Args:
        data_list: A list of dictionaries, where each dictionary has 'id' and
        'parent_id' fields. Assumes there is exactly one item with
        parent_id=None and node_type='submission' (root node).
        max_depth: The maximum depth of the tree to build (optional). Nodes
        deeper than this are not included. If None, there is no depth limit.

    Returns:
        A dictionary representing the root node of the tree, up to the specified
        max_depth, or None if no root is found. The root node has 'id' and
        'tree' (a list of children nodes) fields. Returns None if no root node.
    """

    node_map: dict[str, dict] = {}
    children_map: dict[str, list[dict]] = {}
    root_node = None

    if max_depth == 0:
        return None

    for item in data_list:
        node_id = item.get("id")
        parent_id = item.get("parent_id")

        if node_id is None:
            continue

        # Create a copy of the item to avoid modifying the original data_list in place
        # by the tree structure.
        node = {"id": node_id, "tree": [], **item}
        node_map[node_id] = node

        if parent_id is not None:
            parent_id_stripped = parent_id[3:] # Remove 't1_' or 't3_' prefix
            if parent_id_stripped not in children_map:
                children_map[parent_id_stripped] = []
            children_map[parent_id_stripped].append(node)
        else:
            # The submission is expected to have parent_id=None and node_type='submission'
            if item.get('node_type') == 'submission':
                root_node = node

    if root_node is None:
        # This warning is important if a submission is expected as root.
        print("WARNING: combine_nodes_to_tree did not find an explicit submission root node for this conversation.")
        # Fallback if no explicit submission root is found, maybe a lone comment?
        # For our specific use case, data_list should always include the submission.
        return None

    def build_tree_recursive(nodes: list[dict], current_depth: int):
        if max_depth is not None and current_depth >= max_depth:
            # Nodes beyond max_depth are not included, their children are truncated.
            # We return them but with an empty 'tree' list.
            for node in nodes:
                node["tree"] = []
            return nodes
            
        tree_nodes = []
        for node in nodes:
            children = children_map.get(node["id"], [])
            node["tree"] = build_tree_recursive(children, current_depth + 1)
            tree_nodes.append(node)
        return tree_nodes

    # Start building from the children of the identified root_node.
    # Root nodes (submissions) are considered at depth 0 for internal tracking,
    # so their direct children are at depth 1.
    children_of_root = children_map.get(root_node["id"], [])
    root_node["tree"] = build_tree_recursive(children_of_root, 1)

    return root_node


def count_size_of_tree(x: Dict) -> int:
    """
    Recursively count the size of the tree x.
    Assumes node has a 'tree' key for children.
    """
    return sum([count_size_of_tree(y) for y in x.get("tree", [])]) + 1


def trim_and_get_size(node: dict, depth=0, max_trim_depth=5, trim_branch_factor=2) -> int:
    """
    Trim the tree so that branching factor is limited to `trim_branch_factor`.
    For each node, the "top" `trim_branch_factor` children are selected (and others are ignored).
    We prefer children with greater score. If there's a tie, we prefer
    children with larger (pre-trimmed) tree size.
    Also, trims beyond `max_trim_depth`.
    Modifies the tree in-place.
    Returns the size of the trimmed subtree rooted at 'node'.
    """
    scores_and_children = []  # List of (score, pre_trimmed_size, child_node) for sorting

    # First, recursively process children and collect their scores/sizes
    for child in node.get("tree", []):
        if depth + 1 < max_trim_depth:
            # Recursively trim and get the size of the child's trimmed subtree
            child_trimmed_size = trim_and_get_size(child, depth + 1, max_trim_depth, trim_branch_factor)
            scores_and_children.append((child.get("score", 0), child_trimmed_size, child))
        else:
            # If max_trim_depth is reached, this child's subtree is truncated
            child["tree"] = [] # Clear children beyond max depth
            # Its size is just itself (1)
            scores_and_children.append((child.get("score", 0), 1, child))

    # Sort children based on score then pre_trimmed_size (both descending)
    # The lambda key sorts by score (desc), then by size (desc)
    scores_and_children.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Select top `trim_branch_factor` children
    node["tree"] = [s[2] for s in scores_and_children[:trim_branch_factor]]
    
    # Calculate the size of the current trimmed tree
    new_size = sum([count_size_of_tree(child) for child in node["tree"]]) + 1
    return new_size

def get_flat_nodes_and_edges_from_trimmed_tree(
    node: Dict, 
    collected_nodes: List[Dict], 
    collected_text_community_edges: List[Dict], 
    collected_text_user_edges: List[Dict],
    parent_id: Optional[str] = None,
    required_keys: Optional[List[str]] = None
):
    """
    Recursively collects all nodes and their edges from a trimmed tree into flat lists.
    This is for text_nodes.parquet, text_community_edges.parquet, text_user_edges.parquet.
    """
    # Use all node items if required_keys is not provided, 
    # but filter based on the passed list otherwise.
    if required_keys:
        # Dynamically filter node data based on the passed schema keys
        node_data_for_parquet = {k: node.get(k) for k in required_keys}
    else:
        # Fallback to previous behavior if keys aren't passed (or define a safe default)
        node_data_for_parquet = {k: v for k, v in node.items() if k != 'tree'}
    collected_nodes.append(node_data_for_parquet)

    # Add Text -> Community edge for current node
    if node.get('subreddit'):
        collected_text_community_edges.append({
            'source_id': node['id'],
            'target_id': node['subreddit'],
            'edge_type': 'posts_in' if node.get('node_type') == 'submission' else 'comments_in'
        })

    # Add Text -> User edge for current node
    if node.get('author'):
        collected_text_user_edges.append({
            'source_id': node['id'],
            'target_id': node['author'],
            'edge_type': 'posted_by' if node.get('node_type') == 'submission' else 'commented_by'
        })
    
    # Recursively process children and add parent-child edges
    for child in node.get("tree", []):
        # Add a reply edge between parent and child if the child is a comment
        if child.get('node_type') == 'comment':
            # This is a reply edge from child to parent
            collected_text_user_edges.append({ # Re-using text_user_edges for reply edges (can be split if needed)
                'source_id': child['id'],
                'target_id': node['id'], # Parent node is the target
                'edge_type': 'replies_to'
            })
        
        # Recurse for the child
        get_flat_nodes_and_edges_from_trimmed_tree(
            child, 
            collected_nodes, 
            collected_text_community_edges, 
            collected_text_user_edges,
            node['id'], 
            required_keys
        )