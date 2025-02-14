from flask import Blueprint

finetune_bp = Blueprint('finetune', __name__)


@finetune_bp.route('/finetune', methods=['POST'])
def finetune():
    pass
