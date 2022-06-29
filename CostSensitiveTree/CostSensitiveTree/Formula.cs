using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostSensitiveTree
{
    class Formula
    {
        public string attributeName { get; set; }
        public string symbol { get; set; }
        public double attributeValue_double { get; set; }
        public string attributeValue_string { get; set; }



        public bool IsWeakStronger(Formula f2)
        {
            if (attributeValue_double == Tree.EmptyValue)
                return this.Equals(f2);

            if (attributeName.Equals(f2.attributeName))
            {
                if (f2.symbol.Equals("=") && symbol.Equals("="))
                {
                    if (attributeValue_double == f2.attributeValue_double)
                        return true;
                    else
                        return false;
                }
                else
                {
                    if (f2.symbol.Equals("<"))
                    {
                        if (symbol.Equals("<") && attributeValue_double <= f2.attributeValue_double)
                        {
                            return true;
                        }
                        else
                        {
                            if (symbol.Equals("<=") && attributeValue_double < f2.attributeValue_double)
                            {
                                return true;
                            }
                            else
                            {
                                if (symbol.Equals("=") && attributeValue_double < f2.attributeValue_double)
                                {
                                    return true;
                                }
                                else
                                {
                                    return false;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (f2.symbol.Equals("<="))
                        {
                            if (symbol.Equals("<") && attributeValue_double <= f2.attributeValue_double)
                            {
                                return true;
                            }
                            else
                            {
                                if (symbol.Equals("<=") && attributeValue_double <= f2.attributeValue_double)
                                {
                                    return true;
                                }
                                else
                                {
                                    if (symbol.Equals("=") && attributeValue_double <= f2.attributeValue_double)
                                    {
                                        return true;
                                    }
                                    else
                                    {
                                        return false;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (f2.symbol.Equals(">"))
                            {
                                if (symbol.Equals(">") && attributeValue_double >= f2.attributeValue_double)
                                {
                                    return true;
                                }
                                else
                                {
                                    if (symbol.Equals(">=") && attributeValue_double > f2.attributeValue_double)
                                    {
                                        return true;
                                    }
                                    else
                                    {
                                        if (symbol.Equals("=") && attributeValue_double > f2.attributeValue_double)
                                        {
                                            return true;
                                        }
                                        else
                                        {
                                            return false;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (f2.symbol.Equals(">="))
                                {
                                    if (symbol.Equals(">") && attributeValue_double >= f2.attributeValue_double)
                                    {
                                        return true;
                                    }
                                    else
                                    {
                                        if (symbol.Equals(">=") && attributeValue_double >= f2.attributeValue_double)
                                        {
                                            return true;
                                        }
                                        else
                                        {
                                            if (symbol.Equals("=") && attributeValue_double >= f2.attributeValue_double)
                                            {
                                                return true;
                                            }
                                            else
                                            {
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            }
            else
            {
              //  throw new NotImplementedException();
                return false;
            }

            return false;
        }

        public bool IOppositeFormula(Formula f2)
        {
            if (attributeValue_double == Tree.EmptyValue)
                return this.Equals(f2);

            if (attributeName.Equals(f2.attributeName))
            {
                if (f2.symbol.Equals("=") && symbol.Equals("="))
                {
                    if (attributeValue_double == f2.attributeValue_double)
                        return false;
                    else
                        return true;
                }
                else
                {
                    if (f2.symbol.Equals("<"))
                    {
                        if (symbol.Equals(">") && attributeValue_double >= f2.attributeValue_double)
                        {
                            return true;
                        }
                        else
                        {
                            if (symbol.Equals(">=") && attributeValue_double >= f2.attributeValue_double)
                            {
                                return true;
                            }
                            else
                            {
                                if (symbol.Equals("=") && attributeValue_double >= f2.attributeValue_double)
                                {
                                    return true;
                                }
                                else
                                {
                                    return false;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (f2.symbol.Equals("<="))
                        {
                            if (symbol.Equals(">") && attributeValue_double >= f2.attributeValue_double)
                            {
                                return true;
                            }
                            else
                            {
                                if (symbol.Equals(">=") && attributeValue_double > f2.attributeValue_double)
                                {
                                    return true;
                                }
                                else
                                {
                                    if (symbol.Equals("=") &&  attributeValue_double > f2.attributeValue_double)
                                    {
                                        return true;
                                    }
                                    else
                                    {
                                        return false;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (f2.symbol.Equals(">"))
                            {
                                if (symbol.Equals("<") && attributeValue_double <= f2.attributeValue_double)
                                {
                                    return true;
                                }
                                else
                                {
                                    if (symbol.Equals("<=") && attributeValue_double <= f2.attributeValue_double)
                                    {
                                        return true;
                                    }
                                    else
                                    {
                                        if (symbol.Equals("=") && attributeValue_double <= f2.attributeValue_double)
                                        {
                                            return true;
                                        }
                                        else
                                        {
                                            return false;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (f2.symbol.Equals(">="))
                                {
                                    if (symbol.Equals("<") && attributeValue_double <= f2.attributeValue_double)
                                    {
                                        return true;
                                    }
                                    else
                                    {
                                        if (symbol.Equals("<=") && attributeValue_double < f2.attributeValue_double)
                                        {
                                            return true;
                                        }
                                        else
                                        {
                                            if (symbol.Equals("=") && attributeValue_double < f2.attributeValue_double)
                                            {
                                                return true;
                                            }
                                            else
                                            {
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            }
            throw new NotImplementedException();
        }

        public override bool Equals(object obj)
        {
            if (obj is Formula)
            {
                Formula f2 = (Formula)obj;
                return (f2.attributeName.Equals(attributeName) & f2.attributeValue_double.Equals(attributeValue_double) & f2.attributeValue_string.Equals(attributeValue_string));
            }
            return false;
        }

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        public override string ToString()
        {
            if (attributeValue_string == "")
                return (attributeName + " " + symbol + " " + attributeValue_double);
            else
                return (attributeName + " " + symbol + " " + attributeValue_string);
        }

        public bool IsInverseSign(Formula sign2)
        {
            if (!symbol.Equals(sign2.symbol) && attributeValue_double.Equals(sign2.attributeValue_double) && attributeValue_string.Equals(sign2.attributeValue_string))
                return true;
            return false;
        }
    }
}
