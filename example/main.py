import sys

sys.path.append("./")

import preprocessing
import train
import demo
import plots
from utils.parse_yaml import ParseYaml


def main(yf):
    parser = ParseYaml(yf)

    to_preprocessing = parser.get_item("to_preprocessing")
    to_train = parser.get_item("to_train")
    to_demo = parser.get_item("to_demo")
    to_plots = parser.get_item("to_plots")

    preprocessing_config = parser.get_item("preprocessing")
    if to_preprocessing:
        preprocessing.preprocessing(preprocessing_config)

    train_config = parser.get_item("train")
    if to_train:
        train.train(train_config)

    demo_config = parser.get_item("demo")
    if to_demo:
        demo.demo(demo_config)

    plot_config = parser.get_item("plots")
    if to_plots:
        plots.line_plot(plot_config.get("line_plot"))
        # print("---1---")
        plots.day_scatter_plot(plot_config.get("day_scatter_plot"))
        plots.multi_day_scatter_plot(plot_config.get("multi_day_scatter_plot"))
        # print("---2---")
        plots.hist_plot(plot_config.get("hist_plot"))
        # print("---3---")
        plots.day_map_plot(plot_config.get("day_map_plot"))
        plots.mutli_day_map_plot(plot_config.get("multi_day_map_plot"))


if __name__ == '__main__':
    yf = sys.argv[1]
    main(yf)
