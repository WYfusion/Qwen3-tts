# coding=utf-8

from __future__ import annotations

from .args import build_parser, namespace_to_config
from .trainer import SFTTrainer


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = namespace_to_config(args)
    trainer = SFTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
