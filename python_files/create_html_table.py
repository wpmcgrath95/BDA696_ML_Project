#!/usr/bin/env python3
import os
import re
import sys
import webbrowser


class CreateHTMLTable(object):
    def __init__(self, info_dict, plot_paths):
        self.plot_paths = plot_paths
        self.info_dict = info_dict
        self.num_of_preds = len(self.info_dict["predictor"])

    def print_rows(self, i):
        start = "<tr>\n"

        resp_row = "<td>" + self.info_dict["response"][i] + "</td>\n"
        pred_row = "<td>" + self.info_dict["predictor"][i] + "</td>\n"
        resp_type_row = "<td>" + self.info_dict["response_type"][i] + "</td>\n"
        pred_type_row = "<td>" + self.info_dict["predictor_type"][i] + "</td>\n"
        corr = "<td>" + str(self.info_dict["correlation"][i]) + "</td>\n"

        plot_list = []
        for path in self.plot_paths:
            x = re.findall(self.info_dict["predictor"][i], path)
            if x:
                plot_list.append(path)

        if len(plot_list) > 1:
            p1_row = (
                "<td><a href=" + plot_list[0] + ">" + f"plot {i+1}" + "</a></td>" + "\n"
            )
            p2_row = (
                "<td><a href=" + plot_list[1] + ">" + f"plot {i+1}" + "</a></td>" + "\n"
            )
            plts_row = p1_row + p2_row
        else:
            plts_row = (
                "<td><a href=" + plot_list[0] + ">" + f"plot {i+1}" + "</a></td>" + "\n"
            )

        end = "</tr>\n"
        indent = "</indent>\n"

        one_half = start + resp_row + pred_row + resp_type_row + pred_type_row
        sec_half = corr + plts_row + end + indent

        return one_half + sec_half

    def main(self):
        html_str = ""
        str1 = (
            """
        <html>
            <body>
                <h1>Will's Plot Results</h1>
                <hr/>
                <table border=1>
                    <tr>
                        <th>"""
            + "Response"
            + """</th>
                        <th>"""
            + "Predictor"
            + """</th>
                        <th>"""
            + "Response Type"
            + """</th>
                        <th>"""
            + "Predictor Type"
            + """</th>
                        <th>"""
            + "Correlation"
            + """</th>
                        <th>"""
            + "Plot1 URL"
            + """</th>
                        <th>"""
            + "Plot2 URL"
            + """</th>
                    </tr>
                    <indent>
                    """
        )

        for indx in range(self.num_of_preds):
            holder = self.print_rows(indx)
            html_str = html_str + holder
        str3 = """
                </table>
            </body>
         </html> """

        html_str = str1 + html_str + str3
        try:
            with open("./html_files/html_table.html", "w") as file:
                file.write(html_str)
        except FileNotFoundError:
            print("Error no html_fil directory")
            sys.exit(CreateHTMLTable().main())

        webbrowser.open("file://" + os.path.realpath("./html_files/html_table.html"))


if __name__ == "__main__":
    sys.exit(CreateHTMLTable().main())
