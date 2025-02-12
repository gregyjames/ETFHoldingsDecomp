<div align="center">
<img src="logo.svg" alt="drawing" width="200"/>

# ETFHoldingsDecomp
</div>
Values the top holdings of ETFs to assess if it is overall overvalued or undervalued at the current price. 

### DEMO: [HERE](https://etfholdingsdecomp.streamlit.app/)
### Running Locally
```shell
uv run main.py
```
## Todo
- [x] We are currently using Benjamin Graham's formula to value the holdings, but I don't believe this is suitable in the current Market. A better solution would be to use a discount cash flow model to value the stocks. Just need to find out where to pull this data from.
- [x] Streamlit live dashboard

## License

MIT License

Copyright (c) 2024 Greg James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
