import os, datetime

directory = "random_stocks"
start_date = datetime.datetime.strptime("2009-04-20", "%Y-%m-%d")
end_date = datetime.datetime.strptime("2019-04-20", "%Y-%m-%d")
step = datetime.timedelta(days=1)

if __name__ == "__main__":
    tickers = set()
    valid_dates = set()
    stock_map = {}
    csv_rows = []

    for filename in os.listdir(directory):
        if filename[-3:] == "csv":
            print("Processing {0}".format(filename))

            with open("{0}/{1}".format(directory, filename), "r") as fp:
                fp.readline()

                for line in fp.readlines():
                    date, open_price, _, _, close_price, _, _ = line.split(",")
                    tickers.add(filename.split(".")[0])
                    valid_dates.add(datetime.datetime.strptime(date, "%Y-%m-%d"))
                    stock_map["{0}//{1}".format(filename.split(".")[0], date)] = (float(open_price), float(close_price))

    while start_date <= end_date:
        if start_date in valid_dates:
            csv_row = [start_date.date()]

            for ticker in sorted(tickers):
                line = stock_map["{0}//{1}".format(ticker, start_date.date())]
                csv_row.append(line[0])
                csv_row.append(line[1])

            csv_rows.append(csv_row)

        start_date += step

    header_row = ["date"]

    for ticker in sorted(tickers):
        header_row.append("{0}-open".format(ticker.lower()))
        header_row.append("{0}-close".format(ticker.lower()))

    csv_rows = [header_row] + csv_rows

    with open("{0}.csv".format(directory), "w") as fp:
        for csv_row in csv_rows:
            fp.write(",".join([str(x) for x in csv_row]) + "\n")
