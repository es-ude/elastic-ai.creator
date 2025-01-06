import click
from .hw_test_runner import TestBenchReport, VhdlTestBenchRunner
import xml.etree.ElementTree as ET
import logging
from io import BytesIO
import sys


@click.command()
@click.option(
    "-o",
    "--output-file",
    help="the file that the xml output will be written to, use - for stdout",
    type=click.File(mode="wb"),
)
@click.option(
    "-v", "--verbose", help="wether to print debug output", is_flag=True, default=False
)
def main(verbose, output_file: BytesIO | None):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    report = TestBenchReport()
    test_benches = VhdlTestBenchRunner(report)
    test_benches.run()
    if output_file is not None:
        report.dump(output_file)
    for line in report.pretty_print():
        click.echo(line)

    if report.num_failures() == 0:
        sys.exit(0)
    else:
        sys.exit(1)
