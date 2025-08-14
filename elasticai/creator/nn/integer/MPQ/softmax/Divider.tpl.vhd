library ieee;
use ieee.std_logic_1164.ALL;
use ieee.numeric_std.ALL;
entity ${name} is
   generic ( DATA_WIDTH  : integer range 4 to 32 := ${data_width} );
   port ( enable    : in   std_logic;
      clock     : in   std_logic;
      divisor   : in   std_logic_vector (DATA_WIDTH-1 downto 0);
      dividend  : in   std_logic_vector (DATA_WIDTH-1 downto 0);
      quotient  : out  std_logic_vector (DATA_WIDTH-1 downto 0);
      remainder : out  std_logic_vector (DATA_WIDTH-1 downto 0);
      done      : out  std_logic);
end ${name};
architecture rtl of ${name} is
signal dividend_unsigned  : unsigned(DATA_WIDTH-1 downto 0);
signal divisor_unsigned   : unsigned(DATA_WIDTH-1 downto 0);
signal quotient_unsigned  : unsigned(DATA_WIDTH-1 downto 0);
signal remainder_unsigned : unsigned(DATA_WIDTH-1 downto 0);
signal bits : integer range DATA_WIDTH downto 0;
type states is (idle, prepare, shift, sub, finished);
signal current_state : states;
begin
   process
   variable diff : unsigned(DATA_WIDTH-1 downto 0);
   begin
      wait until rising_edge(clock);
      case current_state is
         when idle =>
            if (enable='1') then
               current_state <= prepare;
               done <= '0';
            end if;
            dividend_unsigned <= unsigned(dividend);
            divisor_unsigned <= unsigned(divisor);
         when prepare =>
            quotient_unsigned    <= (others=>'0');
            remainder_unsigned   <= (others=>'0');
            current_state    <= shift;
            bits <= DATA_WIDTH;
            if (divisor_unsigned=0) then
               quotient_unsigned <= (others=>'1');
               remainder_unsigned <= (others=>'1');
               current_state <= finished;
            elsif (divisor_unsigned>dividend_unsigned) then
               remainder_unsigned <= dividend_unsigned;
               current_state <= finished;
            elsif (divisor_unsigned=dividend_unsigned) then
               quotient_unsigned <= to_unsigned(1,DATA_WIDTH);
               current_state <= finished;
            end if;
         when shift =>
            if ( ( remainder_unsigned( DATA_WIDTH-2 downto 0)&dividend_unsigned(DATA_WIDTH-1)) < divisor_unsigned ) then
               bits <= bits-1;
               remainder_unsigned  <=  remainder_unsigned( DATA_WIDTH-2 downto 0)&dividend_unsigned(DATA_WIDTH-1);
               dividend_unsigned   <= dividend_unsigned(DATA_WIDTH-2 downto 0)&'0';
            else
               current_state    <= sub;
            end if;
         when sub =>
            if (bits>0) then
               remainder_unsigned  <=  remainder_unsigned( DATA_WIDTH-2 downto 0)&dividend_unsigned(DATA_WIDTH-1);
               dividend_unsigned   <= dividend_unsigned(DATA_WIDTH-2 downto 0)&'0';
               diff := ( remainder_unsigned( DATA_WIDTH-2 downto 0)&dividend_unsigned(DATA_WIDTH-1)) - divisor_unsigned;
               if (diff(DATA_WIDTH-1)='0') then
                  quotient_unsigned <= quotient_unsigned(DATA_WIDTH-2 downto 0) & '1';
                  remainder_unsigned <= diff;
               else
                  quotient_unsigned <= quotient_unsigned(DATA_WIDTH-2 downto 0) & '0';
               end if;
               bits <= bits-1;
            else
               current_state    <= finished;
            end if;
         when finished =>
            done <= '1';
            if (enable='0') then
               current_state <= idle;
            end if;
      end case;
   end process;
   quotient  <= std_logic_vector(quotient_unsigned);
   remainder <= std_logic_vector(remainder_unsigned);
end;
